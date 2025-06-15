#pragma once


class BigramLanguageModel : public Model
{
	const EmbeddingPtr token_embedding_table;

public:
	BigramLanguageModel(const int vocab_size) :
		token_embedding_table(Embedding::New(vocab_size,vocab_size,"token_embedding_table"))
	{
		AddLayers(
			{
				token_embedding_table
			});
	}

	tuple<TensorPtr,TensorPtr> Forward(const TensorPtr& idx,const TensorPtr& targets=nullptr) override
	{
		// idx and targets are both (B,T) ternsor of integers.
		TensorPtr logits = token_embedding_table->Forward(idx);
		//cout<<logits->Shape()<<endl;

		// Compute loss only if targets are specified (i.e. training or generation).
		TensorPtr loss;
		if(targets)
		{
			// CrossEntropy only implemented for 2d logits (B,C) but logits here it (B,T,C).
			// Flattening the time B and T dimension to give (BT,C) makes the logits 2d and is ok because 
			// it's still calculating cross entropy for each C (next character prediction).
			//
			// *** "BIGRAM" NOTE***
			// Here each (B,T) sample (input character) is treated independently, the sequence information is squashed out
			// because each timestep prediction independently predicts rows from the embedding layer. There is no
			// layer that communicates between tokens and the model only trains on 1 input character to 1 output character.
			//
		
			// logits (B,T,C) -> (BT,C).
			int B = logits->Shape()[0];
			int T = logits->Shape()[1];
			int C = logits->Shape()[2];
			logits = logits->Reshape({B*T,C});

			// targets (B,T) -> (BT).
			B = targets->Shape()[0];
			T = targets->Shape()[1];
			TensorPtr targets_ = targets->Reshape({B*T});

			loss = CrossEntropyLoss().Forward(logits,targets_);
		}

		return tuple(logits,loss);
	}

	TensorPtr Generate(TensorPtr idx,int max_new_tokens)
	{
		// idx is (B,T) array of indicies in the current context.
		for(int i=0;i<max_new_tokens;++i)
		{
			// get the predicitions.
			auto [logits,loss] = Forward(idx);									// Returns (B,T,C)

			// focus only on the last time step
			logits = logits->Slice({{},{-1},{}});								// becomes (B,C)
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
