#ifndef UPSAMPLE_HPP
#define UPSAMPLE_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <functional>

#include "util.hpp"


/**
 * @brief  Two-dimensional nearest-neighbor upsampling.
 * @description
 *	This implementation leverages the Bresenham approximation approach,
 *	originally devised for approximated line drawing on pixelated output
 *	devices, for the selection of the desired closest input positions.
 *	This selection process is performed independently along each feature map
 *	dimension.
 *	In each dimension, output coordinates [0, O) are mapped back to input
 *	coordinates [0, I). With respect to Bresenham, the output assumes the role
 *	of the x- and the input the role of the y-axis in drawing a line of a slope
 *	smaller than 1. The mappings of the first and last pixels, i.e. (0, 0) and
 *	(O-1, I-1) are used as anchors and are assumed to have no associated
 *	approximation errors. For intermediate mappings (o, i), we seek the nearest
 *	input index as:
 *		i = round( (I-1)/(O-1) * o )
 *	We choose to use rounding up for ties and substitute:
 *		X := O-1
 *		Y := I-1
 *	to yield:
 *		i = round( Y/X * o )
 *
 *	For a sequential stepping through the output indices [0, O-1), the
 *	corresponding sequence of input indices can be tracked along just by
 *	maintaining an approximation error. Initially, e_0 = 0. An iterative step
 *	taking o to o+1 produces the tentative error ê_{j+1} = e_j + Y/X assuming
 *	no increment of the input index. If this tentative error remains below
 *	a half, it is carried over to the next step and the input index is
 *	not changed. Otherwise, the input index is incremented and the error is
 *	discounted accordingly. This yields an iteration step as:
 *
 *		ê_{j+1} = e_j + Y/X
 *		e_{j+1} = ê_{j+1} - (ê_{j+1} < 0.5? 0 : 1)
 *		i_{j+1} = i_j     + (ê_{j+1} < 0.5? 0 : 1)
 *
 *	The fractional representation of the error can be made integer by using:
 *
 *		E_j = 2X*e_j + 2Y - X
 *
 *	yielding:
 *
 *		E_0     = 2Y - X
 *		Ê_{j+1} = E_j     + 2Y
 *		E_{j+1} = Ê_{j+1} - (Ê_{j+1} < 2Y? 0 : 2X)
 *		i_{j+1} = i_j     + (Ê_{j+1} < 2Y? 0 :  1)
 *
 *	The detour through the tentative error can be removed from the computation:
 *
 *		E_0     = 2Y - X
 *		E_{j+1} = E_j + 2Y - (E_j < 0? 0 : 2X)
 *		i_{j+1} = i_j      + (E_j < 0? 0 :  1)
 *
 *	As all E_j are sums of Y and X and only their signedness is relevant for
 *	algorithmic decisions, Y and X can initially be reduced by their greatest
 *	common divisor. Finally, observing that the only possible odd contribution
 *	to E_j comes through the initialization of E_0, it can be truncated away:
 *
 *		E_0     = floor(Y - X/2)
 *		E_{j+1} = E_j + Y - (E_j < 0? 0 : X)
 *		i_{j+1} = i_j     + (E_j < 0? 0 : 1)
 */
template<
	unsigned  HI,	// Height of input feature map
	unsigned  WI,	// Width of input feature map
	unsigned  HO,	// Height of output feature map
	unsigned  WO,	// Width of output feature map
	unsigned  CF,	// Channel Fold
	typename  T
>
void upsample_nn(
	hls::stream<T> &src,
	hls::stream<T> &dst
) {
#pragma HLS pipeline II=1 style=flp
	static_assert(HI <= HO, "Output height cannot be smaller than input dimension.");
	static_assert(WI <= WO, "Output width cannot be smaller than input dimension.");
	static_assert(0 < HI, "Input height must be positive.");
	static_assert(0 < WI, "Input width must be positive.");

	//- Output Tensor Navigation --------------------------------------------
	// Counting to -1 for detecting the end of dimension
	static ModCounter<HO>  ho_cnt;
	static ModCounter<WO>  wo_cnt;
	static ModCounter<CF>  cf_cnt;
#pragma HLS reset variable=ho_cnt
#pragma HLS reset variable=wo_cnt
#pragma HLS reset variable=cf_cnt

	//- Error Tracking along each Dimension ---------------------------------

	// Translate dimensions into endpoint distances and remove common factors
	constexpr unsigned  HX_ = HO-1;
	constexpr unsigned  HY_ = HI-1;
	constexpr unsigned  HD_ = gcd(HX_, HY_);
	constexpr unsigned  HX = HX_/HD_;
	constexpr unsigned  HY = HY_/HD_;

	constexpr unsigned  WX_ = WO-1;
	constexpr unsigned  WY_ = WI-1;
	constexpr unsigned  WD_ = gcd(WX_, WY_);
	constexpr unsigned  WX = WX_/WD_;
	constexpr unsigned  WY = WY_/WD_;

	// Error Initialization Values
	constexpr int  HE0 = HY - (HX+1)/2;
	constexpr int  WE0 = WY - (WX+1)/2;
	static ap_int<1+clog2(std::max(HX-HY, HY))>  he = HE0;	// range: Y-X <= e < Y
	static ap_int<1+clog2(std::max(WX-WY, WY))>  we = WE0;
#pragma HLS reset variable=he
#pragma HLS reset variable=we

	//- Ring Buffer for Upsampling Replay -----------------------------------

	// Write Pointer update delay needed to accommodate memory read-out latency.
	constexpr unsigned  WP_DELAY = 4;
	constexpr unsigned  ADDR_BITS = clog2(WI*CF);	// max retraction: full row
	using  ptr_t = ap_int<1 + ADDR_BITS>;
	static T  buf[1<<ADDR_BITS];
	static ptr_t  wp[WP_DELAY] = { 0, };	// write pointer: delay for rp comparison
	static ptr_t  rp = 0;	// read pointer: bounded by wp
	static ptr_t  fp = 0;	// free pointer: bounds wp for next buffer generation
#pragma HLS reset variable=buf off
#pragma HLS reset variable=rp
#pragma HLS reset variable=fp
#pragma HLS reset variable=wp
#pragma HLS dependence variable=buf inter false
#pragma HLS dependence variable=buf intra false
#pragma HLS array_partition variable=wp complete

	//- Output Buffer Register ----------------------------------------------
	static bool  ovld = false;
	static T     obuf;
#pragma HLS reset variable=ovld
#pragma HLS reset variable=obuf off

	// Update delay pipeline for wp
	for(unsigned  i = WP_DELAY-1; i > 0; i--)  wp[i] = wp[i-1];

	// Read into buffer memory if capacity is available
	if(/* wp <= fp' */ ptr_t(wp[0]-fp) >= 0) {
		T  x;
		if(src.read_nb(x)){
			buf[ap_uint<ADDR_BITS>(wp[0]++)] = x;
			// std::cout << "read: " << x << std::endl;
		}
	}

	// Try to clear output buffer
	if(ovld)  ovld = !dst.write_nb(obuf);

	// Try to refill output buffer
	if(!ovld) {
		obuf = buf[ap_uint<ADDR_BITS>(rp)];

		if(/* rp < wp */ ptr_t(rp-wp[WP_DELAY-1]) < 0) {
			// Determine dimensions that will be replayed upon their next output increment
			bool const  repw = we < 0;
			bool const  reph = he < 0;

			int  rp_inc = 1;
			if(cf_cnt.tick()) {
				// Wrapping from the end to the start of a dimension does simply not touch
				// the corresponding error. Both these points are zero-error anchors.
				if(!wo_cnt.tick()) {
					we += repw? WY : WY-WX;
					if(repw)  rp_inc = 1-CF;	// Replay pixel
				}
				else if(!ho_cnt.tick()) {
					he += reph? HY : HY-HX;
					if(reph)  rp_inc = 1-WI*CF;	// Replay row
				}
			}
			rp += rp_inc;
			if(!reph && !repw)  fp = rp;	// Let free pointer follow up without replay

			ovld = true;
		}
	}

} // upsample_nn()

#endif
