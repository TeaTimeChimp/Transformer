// Speedrun.cpp : experiment harness for training character-level Shakespeare models efficiently.
//
// Usage: Speedrun <mode> [--key value]...
//   modes: gradcheck | causal | profile | train
//
#include "..\AutoGrad\Files.h"
#include "..\AutoGrad\Embedding.h"
#include "..\AutoGrad\CrossEntropyLoss.h"
#include "..\AutoGrad\CategoricalDistribution.h"
#include "..\AutoGrad\ADAM.h"
#include "..\AutoGrad\Muon.h"
#include "..\AutoGrad\Linear.h"
#include "..\AutoGrad\Sequential.h"
#include "..\AutoGrad\Dropout.h"
#include "..\AutoGrad\Profiler.h"
#include "AttentionHead.h"
#include "MultiHeadAttention.h"
#include "FeedForward.h"
#include "AttentionBlock.h"
#include "MultiHeadLanguageModel.h"
#include "Models.h"
#include <set>
#include <map>
#include <chrono>
#include <fstream>
#include <functional>


using namespace std;


// Command line options: --key value pairs.
class Options
{
	map<string,string> _values;

public:
	string mode;

	Options(int argc,char** argv)
	{
		if(argc>1)
			mode = argv[1];
		for(int i=2;i+1<argc;i+=2)
		{
			string key = argv[i];
			if(key.rfind("--",0)==0)
				key = key.substr(2);
			_values[key] = argv[i+1];
		}
	}

	string Str(const string& key,const string& def) const
	{
		auto i = _values.find(key);
		return i==_values.end()?def:i->second;
	}

	int Int(const string& key,const int def) const
	{
		auto i = _values.find(key);
		return i==_values.end()?def:stoi(i->second);
	}

	double Dbl(const string& key,const double def) const
	{
		auto i = _values.find(key);
		return i==_values.end()?def:stod(i->second);
	}
};


// Character-level dataset with a 90/10 train/validation split.
class Dataset
{
public:
	string			text;
	vector<char>	chars;
	map<char,int>	stoi;
	vector<int>		train;
	vector<int>		val;

	Dataset(const string& filename)
	{
		text = AsciiFileReader(filename);
		const set<char> charSet(text.begin(),text.end());
		chars.assign(charSet.begin(),charSet.end());
		for(int i=0;i<(int)chars.size();++i)
			stoi[chars[i]] = i;
		vector<int> data;
		data.reserve(text.size());
		for(auto c:text)
			data.push_back(stoi[c]);
		const size_t n = size_t(0.9*data.size());
		train.assign(data.begin(),data.begin()+n);
		val.assign(data.begin()+n,data.end());
	}

	int VocabSize() const
	{
		return (int)chars.size();
	}

	string Decode(const vector<int>& v) const
	{
		string s;
		for(auto i:v)
			s.push_back(chars[i]);
		return s;
	}

	// Random batch of (x,y) of shape (B,T) from data using the given generator.
	static tuple<TensorPtr,TensorPtr> Batch(const vector<int>& data,const int B,const int T,mt19937& gen)
	{
		uniform_int_distribution<int> dist(0,(int)data.size()-T-2);
		vector<FP> xs(B*T),ys(B*T);
		for(int b=0;b<B;++b)
		{
			const int i = dist(gen);
			for(int t=0;t<T;++t)
			{
				xs[b*T+t] = FP(data[i+t]);
				ys[b*T+t] = FP(data[i+t+1]);
			}
		}
		return {Tensor::New(NDData::New({B,T},xs)),Tensor::New(NDData::New({B,T},ys))};
	}
};


// Common interface for every language model in the experiments.
class LanguageModel
{
public:
	virtual ~LanguageModel() = default;
	virtual Model& Net() = 0;
	virtual int ContextLength() const = 0;									// Max context the model may be given.
	virtual TensorPtr Loss(const TensorPtr& x,const TensorPtr& y) = 0;		// Mean cross entropy over (B,T).
	virtual TensorPtr Logits(const TensorPtr& x) = 0;						// (B,T,V).
	virtual string Describe() const = 0;
};


// The model from the tutorial (MultiHeadLanguageModel).
class TutorialModel : public LanguageModel
{
	MultiHeadLanguageModel	_m;
	const int				_block;
	string					_desc;
public:
	TutorialModel(const int vocab,const Options& o) :
		_m(vocab,o.Int("n_embd",64),o.Int("n_head",4),o.Int("n_layer",2),o.Int("block",64),FP(o.Dbl("dropout",0.0))),
		_block(o.Int("block",64))
	{
		_desc = "tutorial n_embd="+to_string(o.Int("n_embd",64))+" n_head="+to_string(o.Int("n_head",4))+" n_layer="+to_string(o.Int("n_layer",2))+" block="+to_string(_block);
	}
	Model& Net() override { return _m; }
	int ContextLength() const override { return _block; }
	TensorPtr Loss(const TensorPtr& x,const TensorPtr& y) override { return _m.Forward(x,y)[1]; }
	TensorPtr Logits(const TensorPtr& x) override { return _m.Forward(x)[0]; }
	string Describe() const override { return _desc; }
};


// GPT built from fused kernels.
class GptModel : public LanguageModel
{
	GptNet	_net;
	string	_desc;

	static GptConfig Config(const int vocab,const Options& o)
	{
		GptConfig c;
		c.vocab = vocab;
		c.embd = o.Int("n_embd",128);
		c.heads = o.Int("n_head",4);
		c.layers = o.Int("n_layer",4);
		c.block = o.Int("block",128);
		c.mlpMult = o.Int("mlp",4);
		c.tied = o.Int("tied",0)!=0;
		c.rope = o.Int("rope",0)!=0;
		c.relu2 = o.Str("act","gelu")=="relu2";
		c.zeroProj = o.Int("zero_proj",0)!=0;
		c.dropout = FP(o.Dbl("dropout",0.0));
		c.seed = o.Int("init_seed",1);
		return c;
	}

public:
	GptModel(const int vocab,const Options& o) :
		_net(Config(vocab,o))
	{
		const GptConfig c = Config(vocab,o);
		_desc = "gpt n_embd="+to_string(c.embd)+" n_head="+to_string(c.heads)+" n_layer="+to_string(c.layers)+" block="+to_string(c.block)
			+" mlp="+to_string(c.mlpMult)+" tied="+to_string(c.tied)+" rope="+to_string(c.rope)+" act="+(c.relu2?"relu2":"gelu")+" zero_proj="+to_string(c.zeroProj)+" dropout="+to_string(c.dropout);
	}
	Model& Net() override { return _net; }
	int ContextLength() const override { return _net.ContextLength(); }
	TensorPtr Logits(const TensorPtr& x) override { return _net.Forward(x); }
	TensorPtr Loss(const TensorPtr& x,const TensorPtr& y) override { return op::CrossEntropy(_net.Forward(x),y); }
	string Describe() const override { return _desc; }
};


// Character LSTM built from the fused LSTM kernel.
class LstmModel : public LanguageModel
{
	LstmNet	_net;
	string	_desc;
public:
	LstmModel(const int vocab,const Options& o) :
		_net(vocab,o.Int("n_embd",64),o.Int("hidden",256),o.Int("n_layer",1),o.Int("block",128),o.Int("init_seed",1),FP(o.Dbl("dropout",0.0)))
	{
		_desc = "lstm n_embd="+to_string(o.Int("n_embd",64))+" hidden="+to_string(o.Int("hidden",256))+" n_layer="+to_string(o.Int("n_layer",1))
			+" block="+to_string(o.Int("block",128))+" dropout="+to_string(o.Dbl("dropout",0.0));
	}
	Model& Net() override { return _net; }
	int ContextLength() const override { return _net.ContextLength(); }
	TensorPtr Logits(const TensorPtr& x) override { return _net.Forward(x); }
	TensorPtr Loss(const TensorPtr& x,const TensorPtr& y) override { return op::CrossEntropy(_net.Forward(x),y); }
	string Describe() const override { return _desc; }
};


unique_ptr<LanguageModel> MakeModel(const int vocab,const Options& o)
{
	const string name = o.Str("model","tutorial");
	if(name=="tutorial")
		return make_unique<TutorialModel>(vocab,o);
	if(name=="gpt")
		return make_unique<GptModel>(vocab,o);
	if(name=="lstm")
		return make_unique<LstmModel>(vocab,o);
	throw runtime_error("Unknown model '"+name+"'.");
}


int CountParameters(Model& m)
{
	int n = 0;
	for(auto& p:m.GetParameters())
		n += p->Data().Size();
	return n;
}


// Compares backprop gradients against central finite differences for a sample of every parameter.
int GradCheck(const Dataset& ds,const Options& o)
{
	auto lm = MakeModel(ds.VocabSize(),o);
	lm->Net().SetMode(Layer::Mode::Inference);	// No dropout so the function is deterministic.
	mt19937 gen(1234);
	const int T = min(lm->ContextLength(),o.Int("T",8));
	auto [x,y] = Dataset::Batch(ds.train,o.Int("B",4),T,gen);

	TensorPtr loss = lm->Loss(x,y);
	loss->Backward();
	cout<<"loss "<<loss->Item()<<endl;

	const FP eps = FP(o.Dbl("eps",1e-2));
	const int samples = o.Int("samples",6);
	double worst = 0.0;
	int failures = 0;
	int pi = 0;
	for(auto& p:lm->Net().GetParameters())
	{
		const NDArray data = p->Data();
		if(!data.IsContiguous()||!p->Gradient())
		{
			cout<<"param "<<pi++<<" "<<p->Name()<<" skipped (no gradient or non-contiguous)"<<endl;
			continue;
		}
		FP* const raw = data.Raw();
		const FP* const grad = p->Gradient()->Data().Contiguous().Raw();
		uniform_int_distribution<int> pick(0,data.Size()-1);
		double paramWorst = 0.0;
		for(int s=0;s<samples;++s)
		{
			const int i = pick(gen);
			const FP orig = raw[i];
			raw[i] = orig+eps;
			const double lp = lm->Loss(x,y)->Item();
			raw[i] = orig-eps;
			const double lm_ = lm->Loss(x,y)->Item();
			raw[i] = orig;
			const double numeric = (lp-lm_)/(2.0*eps);
			const double analytic = grad[i];
			const double err = fabs(numeric-analytic)/max(1e-3,fabs(numeric)+fabs(analytic));
			paramWorst = max(paramWorst,err);
		}
		const bool ok = paramWorst<0.05;
		if(!ok)
			++failures;
		worst = max(worst,paramWorst);
		cout<<"param "<<pi++<<" "<<p->Name()<<" "<<p->Shape()[0]<<(p->Shape().size()>1?"x"+to_string(p->Shape()[1]):"")
			<<" worst rel err "<<paramWorst<<(ok?"":"   <-- FAIL")<<endl;
	}
	cout<<"gradcheck: "<<failures<<" failing parameter tensors, worst rel err "<<worst<<endl;
	return failures?1:0;
}


// A causal language model's prediction at position t must not depend on tokens after t.
// Perturb the last token of each sequence and check that the logits at earlier positions do not change.
int CausalCheck(const Dataset& ds,const Options& o)
{
	auto lm = MakeModel(ds.VocabSize(),o);
	lm->Net().SetMode(Layer::Mode::Inference);
	mt19937 gen(99);
	const int T = lm->ContextLength();
	auto [x,y] = Dataset::Batch(ds.train,2,T,gen);
	const NDArray a = lm->Logits(x)->Data().Contiguous();

	TensorPtr x2 = Tensor::New(NDData::New(*x->Data()));
	FP* const raw = x2->Data().Raw();
	for(int b=0;b<2;++b)
		raw[b*T+T-1] = FP((int(raw[b*T+T-1])+1)%ds.VocabSize());
	const NDArray c = lm->Logits(x2)->Data().Contiguous();

	const int V = ds.VocabSize();
	double maxDiff = 0.0;
	for(int b=0;b<2;++b)
		for(int t=0;t<T-1;++t)
			for(int v=0;v<V;++v)
				maxDiff = max(maxDiff,(double)fabs(a.Raw()[(b*T+t)*V+v]-c.Raw()[(b*T+t)*V+v]));
	cout<<"causal check: max change in logits at positions before the perturbed token = "<<maxDiff<<endl;
	const bool ok = maxDiff<1e-5;
	cout<<(ok?"PASS: no information flows from the future.":"FAIL: earlier positions can see the future token!")<<endl;
	return ok?0:1;
}


// Mean validation loss over a fixed set of validation batches (identical for every run and model).
double EvalLoss(LanguageModel& lm,const vector<int>& data,const int batches,const int B,const int T)
{
	lm.Net().SetMode(Layer::Mode::Inference);
	mt19937 gen(4242);
	double sum = 0.0;
	for(int i=0;i<batches;++i)
	{
		auto [x,y] = Dataset::Batch(data,B,T,gen);
		sum += lm.Loss(x,y)->Item();
	}
	lm.Net().SetMode(Layer::Mode::Training);
	return sum/batches;
}


string Sample(LanguageModel& lm,const Dataset& ds,const int n,mt19937& gen,const FP temperature=FP(1.0))
{
	lm.Net().SetMode(Layer::Mode::Inference);
	vector<int> ctx = {0};	// Start from '\n'.
	const int V = ds.VocabSize();
	const int T = lm.ContextLength();
	for(int i=0;i<n;++i)
	{
		const int start = max(0,(int)ctx.size()-T);
		const int len = (int)ctx.size()-start;
		vector<FP> in(ctx.begin()+start,ctx.end());
		TensorPtr x = Tensor::New(NDData::New({1,len},in));
		const NDArray logits = lm.Logits(x)->Data().Contiguous();
		const FP* last = logits.Raw()+(len-1)*V;
		vector<double> p(V);
		double mx = -1e30;
		for(int v=0;v<V;++v)
			mx = max(mx,(double)last[v]/temperature);
		for(int v=0;v<V;++v)
			p[v] = exp(last[v]/temperature-mx);
		discrete_distribution<int> d(p.begin(),p.end());
		ctx.push_back(d(gen));
	}
	lm.Net().SetMode(Layer::Mode::Training);
	return ds.Decode(vector<int>(ctx.begin()+1,ctx.end()));
}


// Scales all gradients so their combined L2 norm is at most maxNorm. Returns the norm before clipping.
double ClipGlobalNorm(const vector<TensorPtr>& params,const double maxNorm)
{
#ifdef AUTOGRAD_CUDA
	if(NDConfig::useGpu)
	{
		double sq = 0.0;
		for(auto& p:params)
			if(p->Gradient())
				sq += dev::SumSquares(p->Gradient()->Data().Dev(),p->Data().Size());
		const double norm = sqrt(sq);
		if(maxNorm>0.0&&norm>maxNorm)
			for(auto& p:params)
				if(p->Gradient())
					dev::Scale(p->Gradient()->Data().DevModify(),FP(maxNorm/norm),p->Data().Size());
		return norm;
	}
#endif
	double sq = 0.0;
	for(auto& p:params)
		if(p->Gradient())
		{
			const FP* g = p->Gradient()->Data().Raw();
			for(int i=0,n=p->Data().Size();i<n;++i)
				sq += double(g[i])*g[i];
		}
	const double norm = sqrt(sq);
	if(maxNorm>0.0&&norm>maxNorm)
	{
		const FP scale = FP(maxNorm/norm);
		for(auto& p:params)
			if(p->Gradient())
			{
				FP* g = p->Gradient()->Data().Raw();
				for(int i=0,n=p->Data().Size();i<n;++i)
					g[i] *= scale;
			}
	}
	return norm;
}


int Train(const Dataset& ds,const Options& o,const bool profile)
{
	auto lm = MakeModel(ds.VocabSize(),o);
	const int B = o.Int("batch",32);
	const int T = lm->ContextLength();
	const double lr = o.Dbl("lr",1e-3);
	const double budget = o.Dbl("time",60.0);			// Training seconds (excludes evaluation).
	const double evalEvery = o.Dbl("eval_every",15.0);	// Training seconds between evaluations.
	const int evalBatches = o.Int("eval_batches",8);
	const int maxIters = o.Int("iters",profile?5:INT_MAX);
	const string logName = o.Str("log","");
	const double target = o.Dbl("target",0.0);			// Stop as soon as validation loss reaches this.
	bool reached = false;

	cout<<"model: "<<lm->Describe()<<endl;
	cout<<"parameters: "<<CountParameters(lm->Net())<<"  batch="<<B<<" T="<<T<<" lr="<<lr<<" schedule="<<o.Str("schedule","const")
		<<" warmup="<<o.Int("warmup",0)<<" wd="<<o.Dbl("wd",0.0)<<" clip="<<o.Dbl("clip",0.0)<<" opt="<<o.Str("opt","adam")<<(o.Str("opt","adam")=="muon"?" muon_lr="+to_string(o.Dbl("muon_lr",0.02)):"")<<" budget="<<budget<<"s"<<endl;

	const vector<TensorPtr> params = lm->Net().GetParameters();
	const double clip = o.Dbl("clip",0.0);

	// --opt muon: hidden 2D weight matrices use Muon; embeddings, head, biases and gains use Adam.
	const bool useMuon = o.Str("opt","adam")=="muon";
	const double muonLr = o.Dbl("muon_lr",0.02);
	vector<TensorPtr> adamParams,muonParams;
	for(auto& p:params)
	{
		const string& n = p->Name();
		const bool hidden = p->Shape().size()==2&&n!="tok"&&n!="pos"&&n!="head.W";
		(useMuon&&hidden?muonParams:adamParams).push_back(p);
	}
	unique_ptr<Muon> muon;
	if(!muonParams.empty())
		muon = make_unique<Muon>(muonParams,FP(muonLr));
	ADAM opt(adamParams,FP(lr));
	opt.weightDecay = FP(o.Dbl("wd",0.0));
	const int warmup = o.Int("warmup",0);					// Linear warmup iterations.
	const string schedule = o.Str("schedule","const");		// const | cosine | linear (decay over the time budget).
	const double minFrac = o.Dbl("min_lr",0.1);				// Final LR as a fraction of the peak.
	auto lrAt = [&](const int it,const double seconds)
	{
		double f = 1.0;
		const double progress = min(1.0,seconds/budget);
		if(schedule=="cosine")
			f = minFrac+(1.0-minFrac)*0.5*(1.0+cos(3.14159265358979*progress));
		else if(schedule=="linear")
			f = 1.0-(1.0-minFrac)*progress;
		if(warmup>0&&it<warmup)
			f *= double(it+1)/warmup;
		return lr*f;
	};
	ofstream log;
	if(!logName.empty())
	{
		log.open(logName);
		log<<"iter,train_seconds,tokens,train_loss,val_loss"<<endl;
	}

	mt19937 gen(o.Int("seed",1));
	lm->Net().SetMode(Layer::Mode::Training);
	double trainSeconds = 0.0;
	double nextEval = 0.0;
	double emaLoss = 0.0;
	long long tokens = 0;
	int iter = 0;
	for(;;)
	{
		if(!profile&&(trainSeconds>=nextEval||trainSeconds>=budget||iter>=maxIters))
		{
			const double val = EvalLoss(*lm,ds.val,evalBatches,B,T);
			cout<<"iter "<<iter<<"  t="<<fixed<<setprecision(1)<<trainSeconds<<"s  train(ema) "<<setprecision(4)<<emaLoss<<"  val "<<val<<defaultfloat<<endl;
			if(log.is_open())
				log<<iter<<","<<trainSeconds<<","<<tokens<<","<<emaLoss<<","<<val<<endl;
			nextEval += evalEvery;
			if(target>0.0&&val<=target)
			{
				cout<<"TARGET "<<target<<" reached at t="<<fixed<<setprecision(1)<<trainSeconds<<"s (iter "<<iter<<")"<<defaultfloat<<endl;
				reached = true;
			}
		}
		if(trainSeconds>=budget||iter>=maxIters||reached)
			break;

		auto [x,y] = Dataset::Batch(ds.train,B,T,gen);
		if(profile&&iter==1)
		{
			Profiler::Reset();
			Profiler::Enable(true);		// Skip the first (warm-up) iteration.
		}
		const auto start = chrono::steady_clock::now();
		TensorPtr loss;
		{
			Profiler::Scope s(Profiler::Enabled()?"~step.forward":nullptr);
			loss = lm->Loss(x,y);
		}
		{
			Profiler::Scope s(Profiler::Enabled()?"~step.backward":nullptr);
			loss->Backward();
		}
		{
			Profiler::Scope s(Profiler::Enabled()?"~step.optimizer":nullptr);
			if(clip>0.0)
				ClipGlobalNorm(params,clip);
			const double f = lrAt(iter,trainSeconds)/lr;
			opt.SetLearningRate(FP(lr*f));
			opt.Step();
			if(muon)
			{
				muon->SetLearningRate(FP(muonLr*f));
				muon->Step();
			}
		}
		const double dt = chrono::duration<double>(chrono::steady_clock::now()-start).count();
		trainSeconds += dt;
		tokens += (long long)B*T;
		const double l = loss->Item();
		emaLoss = iter==0?l:0.95*emaLoss+0.05*l;
		if(profile)
			cout<<"iter "<<iter<<" loss "<<l<<" step "<<dt*1000.0<<"ms"<<endl;
		++iter;
	}

	if(profile)
	{
		Profiler::Enable(false);
		Profiler::Print();
		return 0;
	}

	mt19937 sgen(o.Int("sample_seed",7));
	cout<<"---- sample ----"<<endl<<Sample(*lm,ds,o.Int("sample_len",500),sgen)<<endl<<"----------------"<<endl;
	return 0;
}


// Runs one forward/backward on the CPU and on the GPU from identical initial weights and compares loss and gradients.
int DevCheck(const Dataset& ds,const Options& o)
{
	mt19937 gen(77);
	const int T = o.Int("block",64);
	auto [x,y] = Dataset::Batch(ds.train,o.Int("batch",8),T,gen);
	vector<vector<FP>> grads[2];
	double loss[2];
	for(int d=0;d<2;++d)
	{
		NDConfig::useGpu = d==1;
		op::DropoutSeed() = 12345;	// Same dropout masks on both devices.
		auto lm = MakeModel(ds.VocabSize(),o);
		lm->Net().SetMode(Layer::Mode::Training);
		TensorPtr l = lm->Loss(x,y);
		l->Backward();
		loss[d] = l->Item();
		for(auto& p:lm->Net().GetParameters())
		{
			const FP* g = p->Gradient()->Data().Raw();
			grads[d].emplace_back(g,g+p->Data().Size());
		}
	}
	NDConfig::useGpu = false;
	cout<<"loss cpu "<<loss[0]<<" gpu "<<loss[1]<<endl;
	double worst = 0.0;
	for(size_t i=0;i<grads[0].size();++i)
	{
		double num = 0.0,den = 0.0;
		for(size_t j=0;j<grads[0][i].size();++j)
		{
			const double a = grads[0][i][j],b = grads[1][i][j];
			num += (a-b)*(a-b);
			den += a*a;
		}
		const double rel = sqrt(num/(den+1e-30));
		worst = max(worst,rel);
		if(rel>1e-3)
			cout<<"param "<<i<<" relative gradient difference "<<rel<<endl;
	}
	const bool ok = fabs(loss[0]-loss[1])<1e-4&&worst<1e-3;
	cout<<"devcheck: worst relative gradient difference "<<worst<<(ok?"  PASS":"  FAIL")<<endl;
	return ok?0:1;
}


#ifdef AUTOGRAD_CUDA
// GPU GEMM throughput on typical shapes (M,N,K, transA, transB).
int GpuGemmBench()
{
	const int shapes[][5] = {{4096,4096,4096,0,0},{16384,1152,384,0,0},{16384,384,1536,0,0},{384,1536,16384,1,0},{16384,384,1536,0,1},{64,4096,1024,0,0}};
	for(auto& sh:shapes)
	{
		const int M = sh[0],N = sh[1],K = sh[2];
		const bool ta = sh[3]!=0,tb = sh[4]!=0;
		float* A = (float*)dev::Alloc((size_t)M*K*4);
		float* B = (float*)dev::Alloc((size_t)K*N*4);
		float* C = (float*)dev::Alloc((size_t)M*N*4);
		dev::Zero(A,(size_t)M*K); dev::Zero(B,(size_t)K*N);
		const int lda = ta?M:K,ldb = tb?K:N;
		dev::Gemm(ta,tb,M,N,K,1.0f,A,lda,B,ldb,0.0f,C,N);
		dev::Synchronize();
		const int reps = 20;
		const auto start = chrono::steady_clock::now();
		for(int r=0;r<reps;++r)
			dev::Gemm(ta,tb,M,N,K,1.0f,A,lda,B,ldb,0.0f,C,N);
		dev::Synchronize();
		const double sec = chrono::duration<double>(chrono::steady_clock::now()-start).count()/reps;
		cout<<"gpu gemm "<<M<<"x"<<N<<"x"<<K<<(ta?" A^T":"")<<(tb?" B^T":"")<<": "<<(2.0*M*N*K/sec/1e12)<<" TFLOP/s"<<endl;
		dev::Free(A); dev::Free(B); dev::Free(C);
	}
	return 0;
}
#endif


// Checks the packed GEMM against the reference implementation and measures throughput.
int GemmTest(const Options& o)
{
	mt19937 gen(5);
	normal_distribution<double> nd(0.0,1.0);
	int failures = 0;
	const int shapes[][3] = {{1,1,1},{5,7,3},{6,16,1},{13,33,300},{72,256,256},{100,65,384},{257,129,513}};
	for(auto& sh:shapes)
		for(int ta=0;ta<2;++ta)
			for(int tb=0;tb<2;++tb)
			{
				const int M = sh[0],N = sh[1],K = sh[2];
				vector<FP> A((size_t)M*K),B((size_t)K*N),C((size_t)M*N),R((size_t)M*N);
				for(auto& v:A) v = FP(nd(gen));
				for(auto& v:B) v = FP(nd(gen));
				for(size_t i=0;i<C.size();++i) C[i] = R[i] = FP(nd(gen));
				const int lda = ta?M:K, ldb = tb?K:N;
				gemm::Gemm<FP>(ta,tb,M,N,K,FP(0.5),A.data(),lda,B.data(),ldb,FP(2),C.data(),N);
				gemm::Reference<FP>(ta,tb,M,N,K,FP(0.5),A.data(),lda,B.data(),ldb,FP(2),R.data(),N);
				double err = 0.0;
				for(size_t i=0;i<C.size();++i)
					err = max(err,(double)fabs(C[i]-R[i])/(1.0+fabs(R[i])));
				if(err>1e-4)
				{
					++failures;
					cout<<"FAIL M="<<M<<" N="<<N<<" K="<<K<<" ta="<<ta<<" tb="<<tb<<" err="<<err<<endl;
				}
			}
	cout<<"gemm correctness: "<<failures<<" failures"<<endl;

	for(int n:{128,384,1024,2048})
	{
		const int M = o.Int("M",n),N = n,K = n;
		vector<FP> A((size_t)M*K,FP(1)),B((size_t)K*N,FP(1)),C((size_t)M*N);
		gemm::Gemm<FP>(false,false,M,N,K,FP(1),A.data(),K,B.data(),N,FP(0),C.data(),N);
		const int reps = n<=384?50:5;
		const auto start = chrono::steady_clock::now();
		for(int r=0;r<reps;++r)
			gemm::Gemm<FP>(false,false,M,N,K,FP(1),A.data(),K,B.data(),N,FP(0),C.data(),N);
		const double s = chrono::duration<double>(chrono::steady_clock::now()-start).count()/reps;
		cout<<"gemm "<<M<<"x"<<N<<"x"<<K<<": "<<(2.0*M*N*K/s/1e9)<<" GFLOP/s"<<endl;
	}
	return failures?1:0;
}


int main(int argc,char** argv)
{
	try
	{
		const Options o(argc,argv);
		const Dataset ds(o.Str("data","Data\\input.txt"));
		cout<<"dataset: "<<ds.text.size()<<" chars, vocab "<<ds.VocabSize()<<", train "<<ds.train.size()<<", val "<<ds.val.size()<<endl;

		NDConfig::useGpu = o.Str("device","cpu")=="gpu";
#ifdef AUTOGRAD_CUDA
		if(NDConfig::useGpu&&!dev::Available())
			throw runtime_error("No CUDA device.");
#endif
		cout<<"device: "<<(NDConfig::useGpu?"gpu":"cpu")<<endl;
#ifdef AUTOGRAD_CUDA
		if(o.mode=="devcheck")
			return DevCheck(ds,o);
		if(o.mode=="gpugemm")
			return GpuGemmBench();
#endif
		if(o.mode=="gemmtest")
			return GemmTest(o);
		if(o.mode=="gradcheck")
			return GradCheck(ds,o);
		if(o.mode=="causal")
			return CausalCheck(ds,o);
		if(o.mode=="profile")
			return Train(ds,o,true);
		if(o.mode=="train")
			return Train(ds,o,false);
		cout<<"usage: Speedrun gradcheck|causal|profile|train [--key value]..."<<endl;
		return 1;
	}
	catch(const std::exception& e)
	{
		cout<<"error: "<<e.what()<<endl;
	}
	catch(const char* e)
	{
		cout<<"error: "<<e<<endl;
	}
	catch(const string& e)
	{
		cout<<"error: "<<e<<endl;
	}
	return 2;
}
