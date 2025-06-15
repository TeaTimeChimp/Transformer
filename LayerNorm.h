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
		TensorPtr xmean = x->Mean(1,true);
		TensorPtr xvar = x->Var(1,true);
		TensorPtr xhat = x->Sub(xmean)->Div(xvar->Add(_eps)->Sqrt());
		TensorPtr out = _gamma->Mul(xhat)->Add(_beta);
		return out;
	}
};

