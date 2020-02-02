#pragma once

#include <VMUtils/modules.hpp>
#include "array.hpp"

VM_BEGIN_MODULE( cufx )

VM_EXPORT
{
	struct Texture
	{
	private:
		struct TextureImpl
		{
			~Inner() { cudaDestroyTextureObject( _ ); }

			cudaTextureObject_t _;
		};

	public:
		template <typename E, size_t N>
		Texture( ArrayND<E, N> const &arr, cudaTextureDesc const &tex_desc )
		{
			cudaResourceDesc res_desc = {};
			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = arr.get();

			cudaCreateTextureObject( &_->_, &res_desc, &tex_desc, nullptr );
		}

		cudaTextureObject_t get() const { return _->_; }

	private:
		shared_ptr<Inner> _ = make_shared<Inner>();
	};
}

VM_END_MODULE()
