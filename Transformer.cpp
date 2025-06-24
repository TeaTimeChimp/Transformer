// Transformer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "..\AutoGrad\Files.h"
#include "..\AutoGrad\Embedding.h"
#include "..\AutoGrad\CrossEntropyLoss.h"
#include "..\AutoGrad\CategoricalDistribution.h"
#include "..\AutoGrad\ADAM.h"
#include "..\AutoGrad\Linear.h"
#include "..\AutoGrad\Sequential.h"
#include "..\AutoGrad\Dropout.h"
#include "AttentionHead.h"
#include "MultiHeadAttention.h"
#include "FeedForward.h"
#include "AttentionBlock.h"
#include "MultiHeadLanguageModel.h"
#include <set>
#include <map>


using namespace std;


ostream& operator << (ostream& os,const NDShape& shape)
{
	cout<<"[";
	bool first = true;
	for(auto i:shape)
	{
		if(first)
			first = false;
		else
			os<<",";			
		os<<i;
	}
	cout<<"]";
	return os;
}


ostream& operator << (ostream& os,const vector<int>& list)
{
	cout<<"[";
	bool first = true;
	for(auto i:list)
	{
		if(first)
			first = false;
		else
			os<<",";			
		os<<i;
	}
	cout<<"]";
	return os;
}


ostream& operator << (ostream& os,const TensorPtr& tensor)
{
	tensor->Print(os);
	return os;
}


// Vector of values.
TensorPtr Tensor(const std::vector<int>& data)
{
	std::vector<double> r(data.size());
	for(size_t i=0;i<data.size();++i)
		r[i] = (double)data[i];
	return Tensor::New(NDData::New({(int)r.size()},r));
}

vector<int> ToList(const TensorPtr& tensor)
{
	int size = tensor->Shape()[0];
	vector<int> values(size);
	for(int i=0;i<size;++i)
	{
		const double v = tensor->Data()[{i}];
		values[i] = (int)v;
	}
	return values;
}


void print(const TensorPtr& x)
{
	x->Print();
	cout<<endl;
}


void print(const char* s)
{
	cout<<s<<endl;
}


void print(const NDShape& v)
{
	cout<<v<<endl;
}


tuple<TensorPtr,TensorPtr> get_batch(const TensorPtr& data,const int batch_size,const int block_size,const bool debug)
{
	vector<TensorPtr> xs;
	vector<TensorPtr> ys;

	// First batch of test data from Python.
	const vector<int> prnd = 
		{
			508015, 869284, 981146, 674214, 931741, 335388, 871847, 251742, 932266,
			419530, 395126, 365991, 786019, 324685, 646576, 516345, 769099, 133118,
			559674, 169355, 867799, 948995, 479274, 253071, 888871, 648479, 181265,
			637187, 970050, 199238, 510759, 197995
		};

	for(int j=0;j<batch_size;++j)
	{
		const int i = debug?prnd[j]:rnd.NextInt(data->Shape()[0]-block_size-1);		// Random position in the source text.
		xs.emplace_back(data->Slice({{i,i+block_size}}));							// X = Consecutive sequence of 'block_size' characters from random position.
		ys.emplace_back(data->Slice({{i+1,i+block_size+1}}));						// Y = Consecutive sequence of 'block_size' characters from one place right of X.
	}
	
	TensorPtr x = Tensor::Stack(xs);
	TensorPtr y = Tensor::Stack(ys);
	
	return tuple(x,y);
}





class BigramLanguageModelSingleHead : public Model
{
	const int _block_size;
	EmbeddingPtr token_embedding_table;
	EmbeddingPtr position_embedding_table;
	HeadPtr sa_head;
	LinearPtr lm_head;

public:
	BigramLanguageModelSingleHead(const int vocab_size,const int n_embd,const int block_size) :
		_block_size(block_size),
		token_embedding_table(Embedding::New(vocab_size,n_embd,"token_embedding_table")),
		position_embedding_table(Embedding::New(block_size,n_embd,"position_embedding_table")),
		sa_head(Head::New(n_embd,block_size,n_embd,0.0)),						// 1 head with 32-dimensional self-attention.
		lm_head(Linear::New(n_embd,vocab_size,"lm_head"))
	{
	}

	const std::vector<LayerPtr> GetLayers() const override
	{
		return
		{
			token_embedding_table,
			position_embedding_table,
			sa_head,
			lm_head
		};
	}

	const Tensors Forward(const TensorPtr& idx) const override
	{
		// idx and targets are both (B,T) ternsor of integers (targets is optional and only needed for training).
		//print(idx->Shape());
		const int B = idx->Shape()[0];
		const int T = idx->Shape()[1];

		// NOTE! The size of the position embedding table limits the max length of a sequence.
		// The position embedding lookup will fail if the sequence is too long e.g. the original generation routine from 0 to 100 etc.
		_ASSERT(T<=_block_size);
		
		// Select embedding vector for each batch sample/timestep.
		TensorPtr tok_emb = token_embedding_table->Forward(idx);									// (B,T,n_embd)
		//print(tok_emb->Shape());

		// Get positional embeddings corresponding each timestep 0 to T-1.					
		TensorPtr pos_emb = position_embedding_table->Forward(Tensor::New(NDData::Arrange(T)));		// (T,n_embd)
		//print(pos_emb->Shape());
		//print(pos_emb);

		// Add the positional embeddings to the token embeddings to give weight to token positon.
		TensorPtr x = tok_emb->Add(pos_emb);														// (B,T,n_embd)
		//print(x->Shape());

		// Apply one 'head' of self-attention.
		x = sa_head->Forward(x);																	// (B,T,head_size)
		//print(x->Shape());
		
		// Project the embeddings onto vocabulary (the set of characters).
		TensorPtr logits = lm_head->Forward(x);														// (B,T,vocab_size)
		//print(logits->Shape());

		return logits;
	}

	const Tensors Forward(const TensorPtr& idx,const TensorPtr& targets) override
	{
		TensorPtr logits = Forward(idx);

		// Compute loss only if targets are specified (i.e. training or generation).
		TensorPtr loss;
		if(targets)
		{
			// CrossEntropy only implemented for 2d logits (B,C) but logits here it (B,T,C).
			// Flattening the time B and T dimension to give (BT,C) makes the logits 2d and is ok because 
			// it's still calculating cross entropy for each C (next character prediction).
		
			// logits (B,T,C) -> (BT,C).
			const int B = logits->Shape()[0];
			const int T = logits->Shape()[1];
			const int C = logits->Shape()[2];
			logits = logits->Reshape({B*T,C});

			// targets (B,T) -> (BT).
			//B = targets->Shape()[0];
			//T = targets->Shape()[1];
			TensorPtr targets_ = targets->Reshape({B*T});

			loss = CrossEntropyLoss().Forward(logits,targets_);
		}

		return {logits,loss};
	}

	TensorPtr Generate(TensorPtr idx,const int max_new_tokens)
	{
		// idx is (B,T) array of indicies in the current context.
		for(int i=0;i<max_new_tokens;++i)
		{
			// crop idx to the last block_size tokens (to avoid exceeding position_embedding layer size).
			TensorPtr idx_cond = idx->Slice({{},{-_block_size,0}});				// (B,T)

			// get the predicitions.
			TensorPtr logits= Forward(idx_cond);								// (B,T,C)

			// focus only on the last time step
			logits = logits->Slice({{},{-1},{}});								// (B,C)
			//logits->Print();

			// apply softmax to get probabilities.
			TensorPtr probs = logits->Softmax(-1);
			//probs->Print();

			// sample from the distribution.			
			TensorPtr idx_next = CategoricalDistribution(probs,false).Sample();	// (B,1)
			//idx_next->Print();
			
			// Example used a multinomial distribution with would have an extra dimension for trials?
			idx_next = idx_next->Reshape({idx_next->Shape()[0],1});
			//idx_next->Print();

			idx = Tensor::Cat(vector<TensorPtr>{idx,idx_next},1);				// (B,T+1)
			//idx->Print();
		}

		//cout<<idx->Shape()<<endl;
		return idx;
	}
};


double Mean(const vector<TensorPtr>& values)
{
	double count = 0.0;
	double sum = 0.0;
	for(auto t:values)
	{
		sum += t->Mean(false)->Data()[{}];
		++count;
	}
	return sum/count;
}


tuple<double,double> estimate_loss(Model& model,const TensorPtr& train_data,const TensorPtr& val_data,const int batch_size,const int block_size)
{
	const int eval_iters = 200;

	model.SetMode(Layer::Mode::Inference);

	vector<TensorPtr> losses;
	for(int i=0;i<eval_iters;++i)
	{
		auto [X,Y] = get_batch(train_data,batch_size,block_size,false);
		auto [logits,loss] = model(X,Y);
		losses.emplace_back(loss);
	}
	const double training_loss = Mean(losses);

	losses.clear();
	for(int i=0;i<eval_iters;++i)
	{
		auto [X,Y] = get_batch(val_data,batch_size,block_size,false);
		auto [logits,loss] = model(X,Y);
		losses.emplace_back(loss);
	}
	const double validation_loss = Mean(losses);

	return tuple(training_loss,validation_loss);
}


void Aside_TheMathematicalTrickInSelfAttention()
{
	// Adding context from all previous characters.

	const int B = 4;	// Batch
	const int T = 8;	// Time
	const int C = 2;	// Channels

	// Some random sample.
	TensorPtr x = Tensor::New(NDData::RandN({B,T,C}));
	cout<<x->Shape()<<endl;
	print(x);

	// Very weak/lossy average of previous values - loses positional information.
	// Ww eant x[b,t] = mean_{i<=t} x[b,i]
	TensorPtr xbow = Tensor::New(NDData::Zeros({B,T,C}));
	for(int b=0;b<B;++b)
	{
		for(int t=0;t<T;++t)
		{
			// Select all timesteps upto and including t.
			TensorPtr xprev = x->Slice({{b},{0,t+1}});		// [t,C] one C for each timestep 0 through t inclusive.
			//print(xprev);

			// Take the mean over timesteps.
			TensorPtr meanC = xprev->Mean(0,false);			// [C].
			//print(meanC);
			
			// Store the mean [C] at [b,t].
			xbow->Slice({{b},{t}})->Data() = meanC->Data();
			//print(xbow);
		}
	}
	cout<<xbow->Shape()<<endl;
	print(xbow);

	// See how first row of x and xbow are the same, but then xbow differs because it the average over all previous rows in x.
	print(x->Slice({{0}}));
	print(xbow->Slice({{0}}));

	// Using a loop is slow, and this can be achieve using matrix multiplication.

	// This shows how you can 'sum' vertically over rows of 'b' by multipling by 1s.
	TensorPtr a = Tensor::New(NDData::New({3,3},1.0));
	TensorPtr b = Tensor::New(NDData::New({3,2},
		{
			2,7,
			6,4,
			6,5
		}));
	TensorPtr c = a->Dot(b);
	cout<<"a="<<endl;
	print(a);
	cout<<"--"<<endl;
	cout<<"b="<<endl;
	print(b);
	cout<<"--"<<endl;
	cout<<"c="<<endl;
	print(c);


	// ...but if instead of 'a' being a matrix of all ones, it's a lower triangular matrix of ones then you get a partial sum.
	a = Tensor::New(NDData::New({3,3},
		{
			1,0,0,		// Can use the 'tril' method on 'ones' to produce this matrix.
			1,1,0,
			1,1,1
		}));
	b = Tensor::New(NDData::New({3,2},
		{
			2,7,
			6,4,
			6,5
		}));
	c = a->Dot(b);
	cout<<"a="<<endl;
	print(a);
	cout<<"--"<<endl;
	cout<<"b="<<endl;
	print(b);
	cout<<"--"<<endl;
	cout<<"c="<<endl;
	print(c);

	// ...further, if instead of using 1s the rows sum to 1, then you get the average.
	a = Tensor::New(NDData::New({3,3},
		{
			1,0,0,		// Can use the 'tril' method on 'ones' to produce this matrix.
			1,1,0,
			1,1,1
		}));
	a = a->Div(a->Sum(1,true));	// Sum 1s in each row and divide each value in row by result to normalise row.
	b = Tensor::New(NDData::New({3,2},
		{
			2,7,
			6,4,
			6,5
		}));
	c = a->Dot(b);
	cout<<"a="<<endl;
	print(a);
	cout<<"--"<<endl;
	cout<<"b="<<endl;
	print(b);
	cout<<"--"<<endl;
	cout<<"c="<<endl;
	print(c);

	// Now applying this technique to the original example...
	TensorPtr wei = Tensor::New(NDData::New({T,T},1.0))->Tril();
	print(wei);
	wei = wei->Div(wei->Sum(1,true));
	print(wei);
	TensorPtr xbow2 = wei->Dot(x);	// (T,T) @ (B,T,C) -> (B,T,T) @ (B,T,C) = (B,T,C)		Effectively doing a weighted sum where the weights pick out the bits of x we're interested in.
	print(xbow2);

	// Print one sample to eyeball.
	print("x:");
	print(x->Slice({{0}}));
	print("xbow:");
	print(xbow->Slice({{0}}));
	print("xbow2:");
	print(xbow2->Slice({{0}}));

	// Check both methods give the same result.
	_ASSERT(xbow->Data().IsEqualTo(xbow2->Data()));

	// Version 3: use Softmax.
	TensorPtr tril = Tensor::New(NDData::Ones({T,T}).Tril());
	print(tril);
	wei = Tensor::Zeros({T,T});																	// Weights start as 0.
	print(wei);
	wei = wei->MaskedFill(tril->Equal(Tensor::Zeros({T,T})),-numeric_limits<FP>::infinity());	// Sets upper right to -inf masking out tokens from the future.
	print(wei);
	wei = wei->Softmax(-1);																		// Normalizes weights that have not been masked  exp(-inf)=0 exp(0)=1.
	TensorPtr xbow3 = wei->Dot(x);	// (T,T) @ (B,T,C) -> (B,T,T) @ (B,T,C) = (B,T,C)
	print(xbow3);

	// Check both methods give the same result.
	_ASSERT(xbow->Data().IsEqualTo(xbow3->Data()));
}


void Aside_SelfAttention()
{
	// Version 4: self-attention!
	const int B = 4;
	const int T = 8;
	const int C = 32;

	// Random input, batch, time, channels.
	TensorPtr x = Tensor::New(NDData::RandN({B,T,C}));			// (B,T,C)

	// Let's see a single Head perform self-attention.
	const int head_size = 16;
	LinearPtr key = Linear::New(C,head_size,"key",false);				// Key layer maps a token to a 'key'.
	LinearPtr query = Linear::New(C,head_size,"query",false);			// Query layer maps a token to a 'query'.

	// Each token passes *independently* through the key and query layers to produce an abstract key and query for each token.
	TensorPtr k = key->Forward(x);								// (B,T,head_size)
	print(k->Shape());
	TensorPtr q = query->Forward(x);							// (B,T,head_size)
	print(q->Shape());

	// Notionally;
	// 'key'	= I'm this type of token at this position.
	// 'query'	= I'm looking for these types of tokens as these positions.

	// Multiply 'key' and 'query' to compute 'affinity'.		(transpose last 2 dimensions of k to make them compatible i.e. q.rows==k.cols).
	TensorPtr wei = q->Dot(k->Transpose());						// (B,T,head_size) @ (B,T,head_size).T = (B,T,head_size) @ (B,head_size,T) = (B,T,T)
	print(wei->Shape());

	// 'wei' is now the 'affinity' or similarity/attraction between tokens. This is now data dependent and not uniform as before.
	//TensorPtr wei = Tensor::Zeros({T,T});

	// Mask lower left to avoid communication from tokens in later timesteps.
	// In other cases you'd want all tokens to communicate. This here is a 'decoder', predicting the future, rather than an 'encoder' e.g. sentiment classification.
	TensorPtr tril = Tensor::New(NDData::Ones({T,T}))->Tril();
	wei = wei->MaskedFill(tril->Equal(Tensor::Zeros({T,T})),-numeric_limits<FP>::infinity());

	// Normalise rows. Eliminates negatives and rows sum to 1 which can be used as weighting.
	wei = wei->Softmax(-1);

	// Could then multiply weights and input to generate importance of each x...
	//TensorPtr out = wei->Dot(x);	
	//print(out->Shape());

	// ...but instead an abstract 'value' is generated from another linear layer in the same was as 'keys' and 'queries' were generated.
	LinearPtr value = Linear::New(C,head_size,"value",false);
	TensorPtr v = value->Forward(x);
	print(v->Shape());

	// ...and the abstract 'value' multiplied by the attention weights.
	TensorPtr out = wei->Dot(v);
	print(out->Shape());

	// In summary, a token says...
	// 'x'		= Is private to me.
	// 'key'	= Here's what I have.
	// 'query'	= Here's what I'm interested in.
	// 'value'	= This is what I will communicate if your interested in me.

	// Cross-attention - where 'queries' come from x, but 'keys' and 'values' are generated from some other source y.
	// This would be more likely be seen in an encoder-decoder model.
}


void Aside_ScaledDotProductAttention()
{
	// From the original 'Attention Is All You Need' paper, they introduce a scalling factor to 'wei'.
	//									  T
	//									QK
	//		Attention(Q,K,V) = softmax( --- )V
	//									√d
	//									  k
	//
	// Where d_sub_k is the attention head size.

	// This is done to bring the variance of 'wei' back to 1.

	// That is; given two variable with standard normal distributions, when these
	// are multiplied, the variance increases to the size of the last dimension of the variables.
	//
	// e.g.
	TensorPtr k = Tensor::New(NDData::RandN({4,8,16}));
	TensorPtr q = Tensor::New(NDData::RandN({4,8,16}));
	TensorPtr wei = q->Dot(k->Transpose());

	print(k->Var());		// Variance ~1.
	print(q->Var());		// Variance ~1.
	print(wei->Var());		// Variance ~16.

	// ...applying the scaling factor...
	wei = wei->Div(Tensor::New(NDData::New({},sqrt(16))));
	print(wei->Var());		// Variance ~1.

	// Why is this important?
	// It's important because 'wei' is passed to a softmax, and a softmax sharpens extreme values to 1.
	// As the variance increses the softmax tends to one-hot encode the values and this means QK picks out
	// single (fewer) other tokens of interest.

	// Diffuse example (good)...
	print(Tensor::New(NDData::New({5},{0.1,-0.2,0.3,-0.2,0.5}))->Softmax(-1));

	// Extreme example (bad)...
	print(Tensor::New(NDData::New({5},{0.1,-0.2,0.3,-0.2,0.5})*8)->Softmax(-1));

	// ...softmax will eventually converge on a single value, the maximum value.

}

void CheckGradient(const TensorPtr& p,const string& filename,const bool transpose=false)
{
	cout<<p->Name()<<": ";
	TensorPtr expected = Tensor::New(NDData::Load(filename));
	if(transpose)
		expected = expected->Transpose();
	TensorPtr actual = p->Gradient();
	if(actual->IsEqualTo(expected))
		cout<<"Pass"<<endl;
	else
	{
		expected = expected->Transpose();
		if(actual->IsEqualTo(expected))
			cout<<"Pass (TRANSPOSED)"<<endl;
		else
			cout<<"Fail"<<endl;
	}
}


int main()
{
	// Asides/Concepts.
	//Aside_TheMathematicalTrickInSelfAttention();
	//Aside_SelfAttention();
	//Aside_ScaledDotProductAttention();

    // Data downloaded from here...
    //  curl https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt --output e:\temp\input.txt

	// Load the data set.
	const string text = AsciiFileReader("Data\\input.txt");
	cout<<"length of dataset in characters: "<<text.length()<<endl;

	// Create a set of unique characters.
	const set<char> charSet(text.begin(),text.end());
	const vector<char> chars(charSet.begin(),charSet.end());
	const int vocab_size = (int)chars.size();
	cout<<string(chars.begin(),chars.end())<<endl;
	cout<<chars.size()<<endl;

	// Create mapping from characters to integers.
	map<char,int> stoi([&chars](){map<char,int> a;for(int i=0;i<chars.size();++i) a.emplace(chars[i],i); return a;}());
	map<int,char> itos([&chars](){map<int,char> a;for(int i=0;i<chars.size();++i) a.emplace(i,chars[i]); return a;}());;
	auto encode = [&stoi] (const string& s)
	{
		vector<int> i;
		for(auto c:s)
			i.emplace_back(stoi[c]);
		return i;
	};
	auto decode = [&itos] (const vector<int>& lst)
	{
		string s;
		for(auto i:lst)
			s.push_back(itos[i]);
		return s;
	};

	cout<<encode("hii there")<<endl;
	cout<<decode(encode("hii there"))<<endl;

	// Encode the input dataset.
	TensorPtr data = Tensor(encode(text));
	cout<<data->Shape()<<endl;
	cout<<data->Slice({{0,1000}})<<endl;

	// Split data into training and validation sets.
	const int n = int(0.9*data->Shape()[0]);
	const TensorPtr train_data = data->Slice({{0,n}});
	const TensorPtr val_data = data->Slice({{n,0}});

	const int block_size = 8;
	cout<<train_data->Slice({{0,block_size+1}})<<endl;

	// Example x and y values.
	{
		TensorPtr x = train_data->Slice({{0,block_size}});
		TensorPtr y = train_data->Slice({{1,block_size+1}});
		for(int t=0;t<block_size;++t)
		{
			TensorPtr context = x->Slice({{0,t+1}});	// In pytorch this still produces a 1d tensor when t=0 and not a scalar tensor.
			TensorPtr target = y->Slice({{t}});
			cout<<"When input is "<<context<<" the target: "<<target<<endl;
		}
	}

	const int batch_size = 32;

	auto [xb,yb] = get_batch(train_data,batch_size,block_size,true);
	cout<<"inputs:"<<endl;
	cout<<xb->Shape()<<endl;
	cout<<xb<<endl;
	cout<<"targets:"<<endl;
	cout<<yb->Shape()<<endl;
	cout<<yb<<endl;

	cout<<"----"<<endl;

	for(int b=0;b<batch_size;++b)		// batch dimension.
	{
		for(int t=0;t<block_size;++t)	// time dimension.
		{
			TensorPtr context = xb->Slice({{b},{0,t+1}});
			TensorPtr target = yb->Slice({{b},{t}});
			cout<<"when input is "<<context<<" target is: "<<target<<endl;
		}
	}

	cout<<xb<<endl;	// Input to the transformer.

	// Check model is working by using a test sequence and the seeded random initialization of model parameters.
	{
		const int n_embd = 32;
		const int n_head = 4;	// Number of attention heads.
		const int n_layer = 3;	// Number of transformer block layers.
		const FP dropout = 0.2;
		MultiHeadLanguageModel m(vocab_size,n_embd,n_head,n_layer,block_size,dropout);
		m.SetMode(Layer::Mode::Training);
		auto [logits,loss] = m.Forward(xb,yb);
		cout<<logits->Shape()<<endl;
		cout<<"Actual loss is "<<loss<<endl;
		if(!loss->IsEqualTo(Tensor::New(NDData::New({},6.41601))))
		{
			cout<<"Expected loss is 6.41601."<<endl;
			throw;
		}
		loss->Backward();
	}

	{
		// Model needs to be much larger to be any good
		const int batch_size = 64;	// Number of sequences to process in parallel.
		const int block_size = 256;	// Number of characters in the context.
		const int n_embd = 384;		// Size of the embedding vector for each character.
		const int n_head = 6;		// Number of attention heads.
		const int n_layer = 6;		// Number of transformer blocks.
		const FP dropout = 0.2;
		MultiHeadLanguageModel m(vocab_size,n_embd,n_head,n_layer,block_size,dropout);


		// Expected loss if probability of each character was equal. 
		// Probability of each character = 1/65 (65 is vocab).
		// Apply 'log' to avoid very small numbers when probabilities are multiplied.
		// Negative because log will produce a -ve number for inputs between 0 and 1.
		cout<<"Expected uniform loss is "<<-log(1.0/65.0)<<endl;

		m.SetMode(Layer::Mode::Inference);
		cout<<decode(ToList(m.Generate(Tensor::Zeros({1,block_size}),100)->Slice({{0}})))<<endl;

		const int max_iters = 50000;
		const int eval_interval = 500;
		const double learning_rate = 3e-4;

		// create a OyTorch optimizer.
		cout<<"Model has "<<m.GetParameters().size()<<" parameters."<<endl;
		ADAM optimizer(m.GetParameters(),learning_rate);

		for(int iter=0;iter<max_iters;++iter)
		{
			if(iter%eval_interval==0)
			{
				auto[train_loss,val_loss] = estimate_loss(m,train_data,val_data,batch_size,block_size);
				cout<<"step "<<iter<<": train loss "<<train_loss<<", val loss "<<val_loss<<"."<<endl;
				cout<<decode(ToList(m.Generate(Tensor::Zeros({1,block_size}),100)->Slice({{0}})))<<endl;
			}

			// sample a batch of data.
			auto [xb,yb] = get_batch(train_data,batch_size,block_size,false);

			// evaluate the loss.
			m.SetMode(Layer::Mode::Training);
			auto [logits,loss] = m.Forward(xb,yb);
			optimizer.ZeroGrad();
			loss->Backward();
			optimizer.Step();
			cout<<loss<<endl;
		}

		cout<<decode(ToList(m.Generate(Tensor::Zeros({1,1}),500)->Slice({{0}})))<<endl;
	}

	return 0;
}
