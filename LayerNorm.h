#pragma once


typedef std::shared_ptr<class LayerNorm> LayerNormPtr;
class LayerNorm : public Layer
{
	TensorPtr _eps;
	TensorPtr _gamma;
	TensorPtr _beta;

	class P
	{
	};
public:
	static LayerNormPtr New(const int dim)
	{
		return std::make_shared<LayerNorm>(P(),dim);
	}

	LayerNorm(const P&,const int dim,const FP eps=1e-5)
	{
		_eps = Tensor::New(NDData::New({},eps));
		_gamma = Tensor::New(NDData::Ones({dim}),true);
		_beta = Tensor::New(NDData::Zeros({dim}),true);
	}

	const std::vector<TensorPtr> GetParameters() const override
	{
		return {_gamma,_beta};
	}

	const Tensors Forward(const TensorPtr& x) const override
	{
		// Normalise each token's feature vector, i.e. over the last (channel) dimension.
		// Using dim 1 on a (B,T,C) input normalised over time instead, mixing statistics from future tokens into every position.
		// Note: Mean(-1,...) can't be used because KoMean treats -1 as "mean of everything".
		const int dim = (int)x->Shape().size()-1;
		TensorPtr xmean = x->Mean(dim,true);
		TensorPtr xvar = x->Var(dim,true,0);	// Bessel correction=0 (biased variance), as PyTorch's LayerNorm.
		TensorPtr xhat = x->Sub(xmean)->Div(xvar->Add(_eps)->Sqrt());
		TensorPtr out = _gamma->Mul(xhat)->Add(_beta);
		return out;
	}
};

