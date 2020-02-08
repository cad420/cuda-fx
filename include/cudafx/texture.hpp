#pragma once

#include <VMUtils/modules.hpp>
#include <VMUtils/attributes.hpp>
#include "array.hpp"

VM_BEGIN_MODULE( cufx )

VM_EXPORT
{
	struct Texture
	{
	private:
		struct Inner
		{
			~Inner() { cudaDestroyTextureObject( _ ); }

			cudaTextureObject_t _;
		};

	public:
		enum AddressMode
		{
			Wrap = 0,
			Clamp = 1,
			Mirror = 2,
			Border = 3,
		};
		enum FilterMode
		{
			None = 0,
			Linear = 1,
		};
		enum ReadMode
		{
			Raw = 0,
			NormalizedFloat = 1,
		};
		struct Options
		{
			VM_DEFINE_ATTRIBUTE( AddressMode, address_mode ) = AddressMode::Wrap;
			VM_DEFINE_ATTRIBUTE( FilterMode, filter_mode ) = FilterMode::None;
			VM_DEFINE_ATTRIBUTE( ReadMode, read_mode ) = ReadMode::Raw;
			VM_DEFINE_ATTRIBUTE( bool, normalize_coords ) = true;

		public:
			static Options as_array()
			{
				return cufx::Texture::Options{}
				  .set_address_mode( cufx::Texture::AddressMode::Border )
				  .set_filter_mode( cufx::Texture::FilterMode::None )
				  .set_read_mode( cufx::Texture::ReadMode::Raw )
				  .set_normalize_coords( false );
			}
		};

	public:
		template <typename E, size_t N>
		Texture( ArrayND<E, N> const &arr, cudaTextureDesc const &tex_desc )
		{
			cudaResourceDesc res_desc = {};
			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = arr.get();

			if ( auto err = cudaCreateTextureObject( &_->_, &res_desc, &tex_desc, nullptr ) ) {
				throw std::logic_error( cudaGetErrorString( err ) );
			}
		}
		template <typename E, size_t N>
		Texture( ArrayND<E, N> const &arr, Options const &opts = {} ) :
		  Texture( arr, to_texture_desc( opts ) )
		{
		}

		cudaTextureObject_t get() const { return _->_; }

	private:
		cudaTextureDesc to_texture_desc( Options const &opts )
		{
			cudaTextureDesc desc = {};
			switch ( opts.address_mode ) {
			case Wrap: desc.addressMode[ 0 ] = cudaAddressModeWrap; break;
			case Clamp: desc.addressMode[ 0 ] = cudaAddressModeClamp; break;
			case Mirror: desc.addressMode[ 0 ] = cudaAddressModeMirror; break;
			case Border: desc.addressMode[ 0 ] = cudaAddressModeBorder; break;
			}
			desc.addressMode[ 1 ] = desc.addressMode[ 2 ] = desc.addressMode[ 0 ];
			switch ( opts.filter_mode ) {
			case None: desc.filterMode = cudaFilterModePoint; break;
			case Linear: desc.filterMode = cudaFilterModeLinear; break;
			}
			switch ( opts.read_mode ) {
			case Raw: desc.readMode = cudaReadModeElementType; break;
			case NormalizedFloat: desc.readMode = cudaReadModeNormalizedFloat; break;
			}
			desc.normalizedCoords = opts.normalize_coords;
			return desc;
		}

	private:
		shared_ptr<Inner> _ = make_shared<Inner>();
	};
}

VM_END_MODULE()
