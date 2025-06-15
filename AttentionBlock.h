#pragma once

#include "LayerNorm.h"


typedef std::shared_ptr<class Block> BlockPtr;
class Block : public Model
{
	const int head_size;
	const MultiHeadAttentionPtr sa;
	const FeedForwardPtr ffwd;
	const LayerNormPtr ln1;
	const LayerNormPtr ln2;

	class P
	{
	};

public:
	static BlockPtr New(const int n_embd,const int n_head,const int block_size,const FP dropout)
	{
		return std::make_shared<Block>(P(),n_embd,n_head,block_size,dropout);
	}

	Block(const P&,const int n_embd,const int n_head,const int block_size,const FP dropout) :
		head_size(n_embd/n_head),
		sa(MultiHeadAttention::New(n_embd,block_size,head_size,n_head,dropout)),
		ffwd(FeedForward::New(n_embd,dropout)),
		ln1(LayerNorm::New(n_embd)),
		ln2(LayerNorm::New(n_embd))
	{
	}

	const std::vector<LayerPtr> GetLayers() const override
	{
		return
		{
			sa,
			ffwd,
			ln1,
			ln2
		};
	}

	const Tensors Forward(const TensorPtr& x) const override
	{
		TensorPtr z = x->Add(sa->Forward(ln1->Forward(x)));
		z = z->Add(ffwd->Forward(ln2->Forward(z)));
		return z;
	}
};
