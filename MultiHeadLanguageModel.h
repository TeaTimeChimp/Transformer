#pragma once


class MultiHeadLanguageModel : public Model
{
	const int _block_size;
	EmbeddingPtr token_embedding_table;
	EmbeddingPtr position_embedding_table;
	SequentialPtr blocks;
	FeedForwardPtr ffwd;
	LinearPtr lm_head;

public:
	MultiHeadLanguageModel(const int vocab_size,const int n_embd,const int block_size,const FP dropout) :
		_block_size(block_size),
		token_embedding_table(Embedding::New(vocab_size,n_embd,"token_embedding_table")),
		position_embedding_table(Embedding::New(block_size,n_embd,"position_embedding_table")),
		blocks(Sequential::New(
			{
				Block::New(n_embd,4,block_size,dropout),
				Block::New(n_embd,4,block_size,dropout),
				Block::New(n_embd,4,block_size,dropout),
				LayerNorm::New(n_embd)
			})),
		ffwd(FeedForward::New(n_embd,dropout)),
		lm_head(Linear::New(n_embd,vocab_size,"lm_head"))
	{
	}

	const std::vector<LayerPtr> GetLayers() const override
	{
		return
		{
			token_embedding_table,
			position_embedding_table,
			blocks,
			ffwd,
			lm_head
		};
	}

	const Tensors Forward(const TensorPtr& idx) const override
	{
		//print(idx->Shape());
		//print(idx);

		// idx and targets are both (B,T) ternsor of integers (targets is optional and only needed for training).
		const int B = idx->Shape()[0];
		const int T = idx->Shape()[1];

		// NOTE! The size of the position embedding table limits the max length of a sequence.
		// The position embedding lookup will fail if the sequence is too long e.g. the original generation routine from 0 to 100 etc.
		_ASSERT(T<=_block_size);
		
		// Select embedding vector for each batch sample/timestep.
		TensorPtr tok_emb = token_embedding_table->Forward(idx);									// (B,T,n_embd)
		//print(tok_emb);

		// Get positional embeddings corresponding each timestep 0 to T-1.					
		TensorPtr pos_emb = position_embedding_table->Forward(Tensor::New(NDData::Arrange(T)));		// (T,n_embd)
		//print(pos_emb);

		// Add the positional embeddings to the token embeddings to give weight to token positon.
		TensorPtr x = tok_emb->Add(pos_emb);														// (B,T,n_embd)
		//print(x);

		// Apply multiple heads of self-attention.
		x = blocks->Forward(x);																		// (B,T,head_size*num_heads)
		//print(x);

		x = ffwd->Forward(x);
		
		// Project the embeddings onto vocabulary (the set of characters).
		TensorPtr logits = lm_head->Forward(x);														// (B,T,vocab_size)
		//print(logits);

		return logits;
	}

	const Tensors Forward(const TensorPtr& idx,const TensorPtr& targets) override
	{
		TensorPtr logits = Forward(idx);

		// CrossEntropy only implemented for 2d logits (B,C) but logits here it (B,T,C).
		// Flattening the time B and T dimension to give (BT,C) makes the logits 2d and is ok because 
		// it's still calculating cross entropy for each C (next character prediction).
		
		// logits (B,T,C) -> (BT,C).
		const int B = logits->Shape()[0];
		const int T = logits->Shape()[1];
		const int C = logits->Shape()[2];
		logits = logits->Reshape({B*T,C});
		//logits->Print();

		// targets (B,T) -> (BT).
		TensorPtr targets_ = targets->Reshape({B*T});

		// Loss computed from checking the prediction at each timestep.
		TensorPtr loss = CrossEntropyLoss().Forward(logits,targets_);				// aka Negative log-likelyhood.

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
			TensorPtr logits = Forward(idx_cond);								// (B,T,C)

			// focus only on the last time step
			logits = logits->Slice({{},{-1},{}});								// (B,C)

			// apply softmax to get probabilities.
			TensorPtr probs = logits->Softmax(-1);

			// sample from the distribution.			
			TensorPtr idx_next = CategoricalDistribution(probs,false).Sample();	// (B,1)
			
			// Example used a multinomial distribution with would have an extra dimension for trials?
			idx_next = idx_next->Reshape({idx_next->Shape()[0],1});

			idx = Tensor::Cat(std::vector<TensorPtr>{idx,idx_next},1);			// (B,T+1)
		}

		return idx;
	}
};
