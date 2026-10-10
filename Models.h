#pragma once

// Models for the speedrun experiments, built from the fused kernels in AutoGrad\KoFused.h.

#include "..\AutoGrad\KoFused.h"
#include "..\AutoGrad\Model.h"
#include <random>


// Graph helpers - each creates one node backed by a fused kernel.
namespace op
{
	inline TensorPtr Linear(const TensorPtr& x,const TensorPtr& W,const TensorPtr& b=nullptr)
	{
		if(b)
			return Tensor::New(true,{x,W,b},std::make_shared<KoLinear>(true));
		return Tensor::New(true,{x,W},std::make_shared<KoLinear>(false));
	}

	inline TensorPtr LayerNorm(const TensorPtr& x,const TensorPtr& gamma,const TensorPtr& beta)
	{
		return Tensor::New(true,{x,gamma,beta},std::make_shared<KoLayerNorm>());
	}

	inline TensorPtr Gelu(const TensorPtr& x)
	{
		return Tensor::New(true,{x},std::make_shared<KoActivation>(KoActivation::Type::Gelu));
	}

	inline TensorPtr Relu(const TensorPtr& x)
	{
		return Tensor::New(true,{x},std::make_shared<KoActivation>(KoActivation::Type::Relu));
	}

	inline TensorPtr Add(const TensorPtr& a,const TensorPtr& b)
	{
		return Tensor::New(true,{a,b},std::make_shared<KoAddSame>());
	}

	inline TensorPtr CausalAttention(const TensorPtr& qkv,const int heads,const bool rope=false)
	{
		return Tensor::New(true,{qkv},std::make_shared<KoCausalAttention>(heads,rope));
	}

	inline TensorPtr Relu2(const TensorPtr& x)
	{
		return Tensor::New(true,{x},std::make_shared<KoActivation>(KoActivation::Type::Relu2));
	}

	inline TensorPtr LSTM(const TensorPtr& x,const TensorPtr& Wx,const TensorPtr& Wh,const TensorPtr& b)
	{
		return Tensor::New(true,{x,Wx,Wh,b},std::make_shared<KoLSTM>());
	}

	inline TensorPtr CrossEntropy(const TensorPtr& logits,const TensorPtr& targets)
	{
		return Tensor::New(true,{logits},std::make_shared<KoSoftmaxCrossEntropy>(targets->Data()));
	}
}


// Holds a flat list of parameters so a model can expose them through the Model/Layer interface.
class ParameterSet : public Layer
{
	std::vector<TensorPtr> _params;

public:
	TensorPtr Add(const std::string& name,const NDArray& init)
	{
		TensorPtr p = Tensor::New(init,true);
		p->Name(name);
		_params.push_back(p);
		return p;
	}

	TensorPtr Normal(const std::string& name,const NDShape& shape,const double std,std::mt19937& gen)
	{
		std::normal_distribution<double> nd(0.0,std);
		NDArray a = NDData::New(shape);
		FP* r = a.Raw();
		for(int i=0;i<a.Size();++i)
			r[i] = FP(nd(gen));
		return Add(name,a);
	}

	TensorPtr Uniform(const std::string& name,const NDShape& shape,const double range,std::mt19937& gen)
	{
		std::uniform_real_distribution<double> ud(-range,range);
		NDArray a = NDData::New(shape);
		FP* r = a.Raw();
		for(int i=0;i<a.Size();++i)
			r[i] = FP(ud(gen));
		return Add(name,a);
	}

	TensorPtr Constant(const std::string& name,const NDShape& shape,const FP value)
	{
		return Add(name,NDData::New(shape,value));
	}

	const Tensors Forward(const TensorPtr& x) const override
	{
		return x;
	}

	const std::vector<TensorPtr> GetParameters() const override
	{
		return _params;
	}
};
typedef std::shared_ptr<ParameterSet> ParameterSetPtr;


struct GptConfig
{
	int		vocab = 65;
	int		embd = 128;
	int		heads = 4;
	int		layers = 4;
	int		block = 128;
	int		mlpMult = 4;
	bool	tied = false;			// Share the token embedding with the output head.
	bool	rope = false;			// Rotary position embeddings instead of a learned position table.
	bool	relu2 = false;			// ReLU^2 MLP activation instead of GELU.
	bool	zeroProj = false;		// Zero-initialise residual output projections (each block starts as identity).
	int		seed = 1;
};


// GPT-style decoder: token (+position) embeddings, pre-LayerNorm blocks of causal self-attention and an MLP,
// final LayerNorm and a linear head.
class GptNet : public Model
{
	struct Layer_
	{
		TensorPtr ln1g,ln1b,Wqkv,bqkv,Wproj,bproj,ln2g,ln2b,Wfc,bfc,Wout,bout;
	};

	ParameterSetPtr		_p;
	const GptConfig		_c;
	TensorPtr			_tok,_pos,_lnfg,_lnfb,_head,_headb;
	std::vector<Layer_>	_blocks;

public:
	GptNet(const GptConfig& c) :
		_p(std::make_shared<ParameterSet>()),
		_c(c)
	{
		std::mt19937 gen(c.seed);
		const int embd = c.embd;
		const int mlp = c.mlpMult*embd;
		const double std = 0.02;
		const double projStd = c.zeroProj?0.0:0.02/std::sqrt(2.0*c.layers);	// GPT-2: scale residual projections by depth.
		_tok = _p->Normal("tok",{c.vocab,embd},std,gen);
		if(!c.rope)
			_pos = _p->Normal("pos",{c.block,embd},std,gen);
		for(int l=0;l<c.layers;++l)
		{
			const std::string n = "h"+std::to_string(l)+".";
			Layer_ b;
			b.ln1g = _p->Constant(n+"ln1.g",{embd},FP(1));
			b.ln1b = _p->Constant(n+"ln1.b",{embd},FP(0));
			b.Wqkv = _p->Normal(n+"qkv.W",{embd,3*embd},std,gen);
			b.bqkv = _p->Constant(n+"qkv.b",{3*embd},FP(0));
			b.Wproj = _p->Normal(n+"proj.W",{embd,embd},projStd,gen);
			b.bproj = _p->Constant(n+"proj.b",{embd},FP(0));
			b.ln2g = _p->Constant(n+"ln2.g",{embd},FP(1));
			b.ln2b = _p->Constant(n+"ln2.b",{embd},FP(0));
			b.Wfc = _p->Normal(n+"fc.W",{embd,mlp},std,gen);
			b.bfc = _p->Constant(n+"fc.b",{mlp},FP(0));
			b.Wout = _p->Normal(n+"out.W",{mlp,embd},projStd,gen);
			b.bout = _p->Constant(n+"out.b",{embd},FP(0));
			_blocks.push_back(b);
		}
		_lnfg = _p->Constant("lnf.g",{embd},FP(1));
		_lnfb = _p->Constant("lnf.b",{embd},FP(0));
		if(!c.tied)
			_head = _p->Normal("head.W",{embd,c.vocab},std,gen);
		_headb = _p->Constant("head.b",{c.vocab},FP(0));
	}

	const std::vector<LayerPtr> GetLayers() const override
	{
		return {_p};
	}

	int ContextLength() const
	{
		return _c.block;
	}

	const Tensors Forward(const TensorPtr& idx) const override
	{
		const int T = idx->Shape()[1];
		_ASSERT(T<=_c.block);
		TensorPtr x = _tok->IndexSelect(idx);										// (B,T,C)
		if(_pos)
			x = x->Add(T==_c.block?_pos:_pos->Slice({{0,T}}));						// (B,T,C) + (T,C)
		for(const Layer_& b:_blocks)
		{
			TensorPtr h = op::LayerNorm(x,b.ln1g,b.ln1b);
			h = op::Linear(h,b.Wqkv,b.bqkv);										// (B,T,3C)
			h = op::CausalAttention(h,_c.heads,_c.rope);							// (B,T,C)
			h = op::Linear(h,b.Wproj,b.bproj);
			x = op::Add(x,h);
			h = op::LayerNorm(x,b.ln2g,b.ln2b);
			h = op::Linear(h,b.Wfc,b.bfc);
			h = _c.relu2?op::Relu2(h):op::Gelu(h);
			h = op::Linear(h,b.Wout,b.bout);
			x = op::Add(x,h);
		}
		x = op::LayerNorm(x,_lnfg,_lnfb);
		if(_c.tied)
			return op::Linear(x,_tok->Transpose(),_headb);							// Share the embedding matrix.
		return op::Linear(x,_head,_headb);											// (B,T,V)
	}
};


// Character LSTM: embedding -> stacked LSTM layers -> linear head.
class LstmNet : public Model
{
	struct Layer_
	{
		TensorPtr Wx,Wh,b;
	};

	ParameterSetPtr		_p;
	const int			_context;
	TensorPtr			_tok,_head,_headb;
	std::vector<Layer_>	_layers;

public:
	LstmNet(const int vocab,const int embd,const int hidden,const int layers,const int context,const int seed) :
		_p(std::make_shared<ParameterSet>()),
		_context(context)
	{
		std::mt19937 gen(seed);
		_tok = _p->Normal("tok",{vocab,embd},1.0,gen);
		int in = embd;
		const double k = 1.0/std::sqrt((double)hidden);	// PyTorch's default LSTM initialisation range.
		for(int l=0;l<layers;++l)
		{
			const std::string n = "lstm"+std::to_string(l)+".";
			Layer_ L;
			L.Wx = _p->Uniform(n+"Wx",{in,4*hidden},k,gen);
			L.Wh = _p->Uniform(n+"Wh",{hidden,4*hidden},k,gen);
			L.b = _p->Constant(n+"b",{4*hidden},FP(0));
			FP* b = L.b->Data().Raw();
			for(int j=hidden;j<2*hidden;++j)
				b[j] = FP(1);								// Forget-gate bias of 1: remember by default.
			_layers.push_back(L);
			in = hidden;
		}
		_head = _p->Uniform("head.W",{hidden,vocab},k,gen);
		_headb = _p->Constant("head.b",{vocab},FP(0));
	}

	const std::vector<LayerPtr> GetLayers() const override
	{
		return {_p};
	}

	int ContextLength() const
	{
		return _context;
	}

	const Tensors Forward(const TensorPtr& idx) const override
	{
		TensorPtr x = _tok->IndexSelect(idx);				// (B,T,E)
		for(const Layer_& L:_layers)
			x = op::LSTM(x,L.Wx,L.Wh,L.b);					// (B,T,H)
		return op::Linear(x,_head,_headb);					// (B,T,V)
	}
};
