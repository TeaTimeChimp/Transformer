#pragma once


typedef std::shared_ptr<class FeedForward> FeedForwardPtr;
class FeedForward : public Model
{
	const FP		_dropout;
	SequentialPtr	net;

	class P
	{
	};
public:
	static FeedForwardPtr New(const int n_embd,const FP dropout)
	{
		return std::make_shared<FeedForward>(P(),n_embd,dropout);
	}

	FeedForward(const P&,const int n_embd,const FP dropout) :
		_dropout(dropout)
	{
		net = Sequential::New(
			{
				Linear::New(n_embd,n_embd*4,"FFWD",true,Relu::New()),
				Linear::New(n_embd*4,n_embd),
				Dropout::New(_dropout)
			});
	}

	const std::vector<LayerPtr> GetLayers() const override
	{
		return {net};
	}

	const Tensors Forward(const TensorPtr& x) const override
	{
		return net->Forward(x);
	}
};
