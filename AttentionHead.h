#pragma once


typedef std::shared_ptr<class Head> HeadPtr;
class Head : public Model
{
	const int			head_no;
	const TensorPtr		itril;	// Inverse of tril, upper right triangle.
	const LinearPtr		key;
	const LinearPtr		query;
	const LinearPtr		value;
	const DropoutPtr	_dropout;

	static int shead_no;

	class P
	{
	};

public:
	static HeadPtr New(const int n_embd,const int block_size,const int head_size,const FP dropout)
	{
		return std::make_shared<Head>(P(),n_embd,block_size,head_size,dropout);
	}

	Head(const P&,const int n_embd,const int block_size,const int head_size,const FP dropout) :
		head_no(shead_no++),
		itril(Tensor::New(NDData::Ones({block_size,block_size}).Tril())->Equal(Tensor::Zeros({block_size,block_size}))),
		key(Linear::New(n_embd,head_size,"key_"+std::to_string(head_no),false)),
		query(Linear::New(n_embd,head_size,"query_"+std::to_string(head_no),false)),
		value(Linear::New(n_embd,head_size,"value_"+std::to_string(head_no),false)),
		_dropout(Dropout::New(dropout))
	{
	}

	const std::vector<LayerPtr> GetLayers() const override
	{
		return 
		{
			key,
			query,
			value,
			_dropout
		};
	}

	const Tensors Forward(const TensorPtr& x) const override
	{
		const int B = x->Shape()[0];																// (B,T,C)	C = channels (set by n_embd)
		const int T = x->Shape()[1];
		//const int C = x->Shape()[2];

		// Compute 'key' and 'query' given x.
		TensorPtr k = key->Forward(x);																// (B,T,H)	H = head_size
		TensorPtr q = query->Forward(x);															// (B,T,H)
		//print(k->Shape());
		//print(k);
		//print(q->Shape());
		//print(q);

		// Compute attention scores ("affinities") - the weights for importance.
		// Each token gets to interact with all tokens in previous timesteps (actually all timesteps tbut the future token are about to be zeroed).
		TensorPtr wei = q->Dot(k->Transpose());														// (B,T,H) @ (B,H,T) = (B,T,T)
		//wei->Print();
		
		// Scale values to bring variance back to 1 to avoid excessive sharpening by softmax.
		// This is done by multiplying by the square root of the number of inputs.
		wei = wei->Mul(Tensor::New(NDData::New({},pow(k->Shape()[2],-0.5))));						// (B,T,T)
		//print(wei->Shape());
		//wei->Print();

		// Mask lower left to avoid communication from tokens in later timesteps.
		wei = wei->MaskedFill(itril,-std::numeric_limits<FP>::infinity());							// (B,T,T)
		//wei->Print();

		// Normalise rows. Eliminates negatives and rows sum to 1 which can be used as weighting.
		// The resulting (B,T,T), for each sample, there's a matrix, with 1 row for each token
		// at each timestep which contains the importance of each previous token.
		wei = wei->Softmax(-1);																		// (B,T,T)
		//wei->Print();

		wei = _dropout->Forward(wei);

		// Perform the weighted aggregation of the input tokens.
		TensorPtr v = value->Forward(x);															// (B,T,H)
		//print(v);

		TensorPtr out = wei->Dot(v);																// (B,T,H)
		//print(out);

		return out;
	}
};
int Head::shead_no = 0;
