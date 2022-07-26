/******************************************************************************
 *  Copyright (c) 2022, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
 
/******************************************************************************
 *
 *  Authors: Yaman Umuroglu <yamanu@amd.com>
 *
 *  @file eltwise.hpp
 *
 * Templatized streaming elementwise operator, consuming two streams and
 * producing a third. The operator and datatypes can be customized.
 *
 ******************************************************************************/

#ifndef ELTWISE_HPP
#define ELTWISE_HPP

 /*!
 * \brief EltwiseFunction: General contract for elementwise functions.
 *
 * This class itself has no formal significance for the implementation
 * of the elementwise function. It provides a guidence for specific elementwise function to be used in 
 * StreamingEltwise
 * 
 * \tparam TIn0 Datatype of the first input to the eltwise function
 * \tparam TIn1 Datatype of the second input to the eltwise function
 * \tparam TOut Datatype of the output generated by the eltwise function
 *
 */
template<typename TIn0, typename TIn1, typename TOut>
class EltwiseFunction {
public:
/*!
 * \brief eltwise_op: computes the eltwise algorithm on a pair of elements
 *
 * \param input0 First input value to be used in the eltwise function 
 * \param input1 Second input value to be used in the eltwise function 
*/
    TOut eltwise_op(TIn0 const &input0, TIn1 const &input1) const;
};

template<typename TIn0, typename TIn1, typename TOut>
class AddEltwiseFunction : public EltwiseFunction<TIn0, TIn1, TOut> {
public:
    TOut eltwise_op(TIn0 const &input0, TIn1 const &input1) const {
#pragma HLS inline
        return (TOut)(input0 + input1);
    }
};

template<typename TIn0, typename TIn1, typename TOut>
class SubEltwiseFunction : public EltwiseFunction<TIn0, TIn1, TOut> {
public:
    TOut eltwise_op(TIn0 const &input0, TIn1 const &input1) const {
#pragma HLS inline
        return (TOut)(input0 - input1);
    }
};

template<typename TIn0, typename TIn1, typename TOut>
class AbsDiffEltwiseFunction : public EltwiseFunction<TIn0, TIn1, TOut> {
public:
    TOut eltwise_op(TIn0 const &input0, TIn1 const &input1) const {
#pragma HLS inline
        return (TOut)((input0 > input1) ? (input0-input1) : (input1-input0));
    }
};

/**
 * \brief StreamingEltwise function
 *
 * The function performs a generic eltwise function on two streams and
 * produces an output stream.
 *
 * \tparam Channels   Number of channels for eltwise operation
 * \tparam PE         Number of channels for eltwise operation computed in parallel
 * \tparam N          Total number of elements to process
 * \tparam SliceIn0   Data slicer for input 0 type
 * \tparam SliceIn1   Data slicer for input 1 type
 * \tparam SliceOut   Data slicer for output type
 * \tparam TStrmIn0   Type of the input 0 stream - safely deducible from the paramaters
 * \tparam TStrmIn1   Type of the input 1 stream - safely deducible from the paramaters
 * \tparam TStrmOut   Type of the output - safely deducible from the paramaters
 * \tparam TFxn       Type of the function class (e.g. Max, Avg, Sum) - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param function    Function to apply, derived from EltwiseFunction
 */
template<
  unsigned Channels, unsigned PE, unsigned N,
  typename SliceIn0,typename SliceIn1, typename SliceOut,
  typename TStrmIn0, typename TStrmIn1, typename TStrmOut,
  typename TFxn
>
void StreamingEltwise(hls::stream<TStrmIn0> &in0,
    hls::stream<TStrmIn1> &in1,
    hls::stream<TStrmOut> &out,
    TFxn const &function) {
  constexpr unsigned  TOTAL_FOLD = (Channels / PE) * N;
  // everything merged into a common iteration space (one big loop instead
  // of smaller nested loops) to get the pipelining the way we want
  for(unsigned  i = 0; i < TOTAL_FOLD; i++) {
#pragma HLS pipeline style=flp II=1
    TStrmIn0 in0_slice = in0.read();
    TStrmIn1 in1_slice = in1.read();
    auto outElem = SliceOut().template operator()<TStrmOut>();
    auto const in0_slice_channels = SliceIn0()(in0_slice,0);
    auto const in1_slice_channels = SliceIn1()(in1_slice,0);
    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        outElem(pe,0,1) = function.eltwise_op(in0_slice_channels(pe,0), in1_slice_channels(pe,0));
    }
    out.write(outElem);
  }
};

#endif