/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#include <stdio.h>
#include <math.h>
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold( float* reference, float* idata, const unsigned int len) 
{
  reference[0] = 0;
  double total_sum = 0;
  unsigned int i;
  for( i = 1; i < len; ++i) 
  {
      total_sum += idata[i-1];
      reference[i] = idata[i-1] + reference[i-1];
  }
  // Here it should be okay to use != because we have integer values
  // in a range where float can be exactly represented
  if (total_sum != reference[i-1])
      printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
  
}

