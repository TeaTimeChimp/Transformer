#pragma once


typedef std::shared_ptr<class MultiHeadAttention> MultiHeadAttentionPtr;
class MultiHeadAttention : public Model
{
	std::vector<HeadPtr> _heads;
	LinearPtr proj;
	DropoutPtr _dropout;

	class P
	{
	};

public:
	static MultiHeadAttentionPtr New(const int n_embd,const int block_size,const int head_size,const int num_heads,const FP dropout)
	{
		return std::make_shared<MultiHeadAttention>(P(),n_embd,block_size,head_size,num_heads,dropout);
	}

	MultiHeadAttention(const P&,const int n_embd,const int block_size,const int head_size,const int num_heads,const FP dropout)
	{
		for(int i=0;i<num_heads;++i)
		{
			const HeadPtr head = Head::New(n_embd,block_size,head_size,dropout);
			_heads.emplace_back(head);
		}
		proj = Linear::New(n_embd,n_embd,"Projection",true);
		_dropout = Dropout::New(dropout);
	}

	const std::vector<LayerPtr> GetLayers() const override
	{
		std::vector<LayerPtr> layers(_heads.begin(),_heads.end());
		layers.emplace_back(proj);
		layers.emplace_back(_dropout);
		return layers;
	}

	const Tensors Forward(const TensorPtr& x) const override
	{
		// Pass x through each head independently.
		std::vector<TensorPtr> y(_heads.size());
		/*
		for(int i=0;i<_heads.size();++i)
			y[i] = _heads[i]->Forward(x);
		*/
		NDThreadPool::ForEach(0,(int)_heads.size(),[&](const int i)
		{
			y[i] = _heads[i]->Forward(x);
		});

		// Concatenate head outputs.
		TensorPtr out = Tensor::Cat(y,-1);

		// Forward through projection layer.
		out = proj->Forward(out);

		// Apply dropout.
		out = _dropout->Forward(out);

		return out;
	}
};
