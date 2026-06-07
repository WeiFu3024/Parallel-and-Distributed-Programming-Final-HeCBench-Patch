//========================================================================================================================================================================================================200
//  MERGED SINGLE-FILE main.cu for heartwall-cuda
//  Sources merged: main.h, kernel/kernel.h, kernel/kernel.cu, main.cu
//  Separately compiled: util/avi/avilib.c, util/avi/avimod.c,
//                       util/file/file.c, util/timer/timer.c
//========================================================================================================================================================================================================200

//========================================================================================================================================================================================================200
//  SYSTEM INCLUDES
//========================================================================================================================================================================================================200

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

//========================================================================================================================================================================================================200
//  UTILITY HEADERS (separately compiled, but we include their headers)
//  Note: timer.h, avilib.h, avimod.h do not depend on fp, so include them early.
//  file.h uses fp in its prototype, so it must come after #define fp float below.
//========================================================================================================================================================================================================200

#include "timer.h"
#include "avilib.h"
#include "avimod.h"

//========================================================================================================================================================================================================200
//  INLINED: main.h
//========================================================================================================================================================================================================200

#define fp float

// file.h uses fp in its prototype, so include it after fp is defined
#include "file.h"

#ifdef RD_WG_SIZE_0_0
        #define NUMBER_THREADS RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define NUMBER_THREADS RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define NUMBER_THREADS RD_WG_SIZE
#else
        #define NUMBER_THREADS 256
#endif

#define CHECK 37

// need to define these for static allocation of constant memory in GPU, because it cannot be allocated dynamically
#define ENDO_POINTS 20
#define EPI_POINTS 31
#define ALL_POINTS 51

//========================================================================================================================================================================================================200
//  PARAMS_COMMON STRUCTURE
//========================================================================================================================================================================================================200

typedef struct params_common{

	//======================================================================================================================================================150
	//	HARDCODED INPUTS FROM MATLAB
	//======================================================================================================================================================150

	//====================================================================================================100
	//	UNIQUE PARAMETER STRUCTURE MEMORY SIZE
	//====================================================================================================100

	int common_change_mem;
	int common_mem;
	int unique_mem;

	//====================================================================================================100
	//	USER INPUT
	//====================================================================================================100

	int frames_processed;

	//====================================================================================================100
	//	CONSTANTS
	//====================================================================================================100

	int sSize;
	int tSize;
	int maxMove;
	fp alpha;

	//====================================================================================================100
	//	FRAME
	//====================================================================================================100

	int no_frames;
	int frame_rows;
	int frame_cols;
	int frame_elem;
	int frame_mem;

	//====================================================================================================100
	//	ENDO POINTS
	//====================================================================================================100

	int endoPoints;
	int endo_mem;

	//====================================================================================================100
	//	EPI POINTS
	//====================================================================================================100
	int epiPoints;
	int epi_mem;

	//====================================================================================================100
	//	ALL POINTS
	//====================================================================================================100

	int allPoints;

	//====================================================================================================100
	//	End
	//====================================================================================================100

	//======================================================================================================================================================150
	//	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY
	//======================================================================================================================================================150

	int in_rows;
	int in_cols;
	int in_elem;
	int in_mem;

	//======================================================================================================================================================150
	//	IN_POINTER
	//======================================================================================================================================================150

	int in_pointer_mem;

	//======================================================================================================================================================150
	// 	AREA AROUND POINT		FROM	FRAME
	//======================================================================================================================================================150

	int in2_rows;
	int in2_cols;
	int in2_elem;
	int in2_mem;

	//======================================================================================================================================================150
	//	CONVOLUTION
	//======================================================================================================================================================150

	int conv_rows;
	int conv_cols;
	int conv_elem;
	int conv_mem;
	int ioffset;
	int joffset;

	//======================================================================================================================================================150
	//	CUMULATIVE SUM 1
	//======================================================================================================================================================150

	//====================================================================================================100
	//	PAD ARRAY, VERTICAL CUMULATIVE SUM
	//====================================================================================================100

	int in2_pad_add_rows;
	int in2_pad_add_cols;
	int in2_pad_cumv_rows;
	int in2_pad_cumv_cols;
	int in2_pad_cumv_elem;
	int in2_pad_cumv_mem;

	//====================================================================================================100
	//	SELECTION
	//====================================================================================================100

	int in2_pad_cumv_sel_rows;
	int in2_pad_cumv_sel_cols;
	int in2_pad_cumv_sel_elem;
	int in2_pad_cumv_sel_mem;
	int in2_pad_cumv_sel_rowlow;
	int in2_pad_cumv_sel_rowhig;
	int in2_pad_cumv_sel_collow;
	int in2_pad_cumv_sel_colhig;

	//====================================================================================================100
	//	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//====================================================================================================100

	int in2_pad_cumv_sel2_rowlow;
	int in2_pad_cumv_sel2_rowhig;
	int in2_pad_cumv_sel2_collow;
	int in2_pad_cumv_sel2_colhig;
	int in2_sub_cumh_rows;
	int in2_sub_cumh_cols;
	int in2_sub_cumh_elem;
	int in2_sub_cumh_mem;

	//====================================================================================================100
	//	SELECTION
	//====================================================================================================100

	int in2_sub_cumh_sel_rows;
	int in2_sub_cumh_sel_cols;
	int in2_sub_cumh_sel_elem;
	int in2_sub_cumh_sel_mem;
	int in2_sub_cumh_sel_rowlow;
	int in2_sub_cumh_sel_rowhig;
	int in2_sub_cumh_sel_collow;
	int in2_sub_cumh_sel_colhig;

	//====================================================================================================100
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================100

	int in2_sub_cumh_sel2_rowlow;
	int in2_sub_cumh_sel2_rowhig;
	int in2_sub_cumh_sel2_collow;
	int in2_sub_cumh_sel2_colhig;
	int in2_sub2_rows;
	int in2_sub2_cols;
	int in2_sub2_elem;
	int in2_sub2_mem;

	//====================================================================================================100
	//	End
	//====================================================================================================100

	//======================================================================================================================================================150
	//	CUMULATIVE SUM 2
	//======================================================================================================================================================150

	//====================================================================================================100
	//	MULTIPLICATION
	//====================================================================================================100

	int in2_sqr_rows;
	int in2_sqr_cols;
	int in2_sqr_elem;
	int in2_sqr_mem;

	//====================================================================================================100
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================100

	int in2_sqr_sub2_rows;
	int in2_sqr_sub2_cols;
	int in2_sqr_sub2_elem;
	int in2_sqr_sub2_mem;

	//====================================================================================================100
	//	End
	//====================================================================================================100

	//======================================================================================================================================================150
	//	FINAL
	//======================================================================================================================================================150

	int in_sqr_rows;
	int in_sqr_cols;
	int in_sqr_elem;
	int in_sqr_mem;

	//======================================================================================================================================================150
	//	TEMPLATE MASK CREATE
	//======================================================================================================================================================150

	int tMask_rows;
	int tMask_cols;
	int tMask_elem;
	int tMask_mem;

	//======================================================================================================================================================150
	//	POINT MASK INITIALIZE
	//======================================================================================================================================================150

	int mask_rows;
	int mask_cols;
	int mask_elem;
	int mask_mem;

	//======================================================================================================================================================150
	//	MASK CONVOLUTION
	//======================================================================================================================================================150

	int mask_conv_rows;
	int mask_conv_cols;
	int mask_conv_elem;
	int mask_conv_mem;
	int mask_conv_ioffset;
	int mask_conv_joffset;

	//======================================================================================================================================================150
	//	End
	//======================================================================================================================================================150

} params_common;

//========================================================================================================================================================================================================200
//  INLINED: kernel/kernel.h  (the __global__ hw kernel)
//========================================================================================================================================================================================================200

__global__ void hw (
  const int frame_no,
  const params_common d_common,
  const fp* __restrict__ d_frame,
  int* __restrict__ d_endoRow,
  int* __restrict__ d_endoCol,
  int* __restrict__ d_tEndoRowLoc,
  int* __restrict__ d_tEndoColLoc,
  int* __restrict__ d_epiRow,
  int* __restrict__ d_epiCol,
  int* __restrict__ d_tEpiRowLoc,
  int* __restrict__ d_tEpiColLoc,
  fp* __restrict__ d_endoT,
  fp* __restrict__ d_epiT,
  fp* __restrict__ d_in2,
  fp* __restrict__ d_conv,
  fp* __restrict__ d_in2_pad_cumv,
  fp* __restrict__ d_in2_pad_cumv_sel,
  fp* __restrict__ d_in2_sub_cumh,
  fp* __restrict__ d_in2_sub_cumh_sel,
  fp* __restrict__ d_in2_sub2,
  fp* __restrict__ d_in2_sqr,
  fp* __restrict__ d_in2_sqr_sub2,
  fp* __restrict__ d_in_sqr,
  fp* __restrict__ d_tMask,
  fp* __restrict__ d_mask_conv,
  fp* __restrict__ d_in_mod_temp,
  fp* __restrict__ d_in_partial_sum,
  fp* __restrict__ d_in_sqr_partial_sum,
  fp* __restrict__ d_par_max_val,
  fp* __restrict__ d_par_max_coo,
  fp* __restrict__ d_in_final_sum,
  fp* __restrict__ d_in_sqr_final_sum,
  fp* __restrict__ d_denomT
#ifdef TEST_CHECKSUM
  , fp* __restrict__ checksum
#endif
)
{
	// __global fp* d_in;
	int rot_row;
	int rot_col;
	int in2_rowlow;
	int in2_collow;
	int ic;
	int jc;
	int jp1;
	int ja1, ja2;
	int ip1;
	int ia1, ia2;
	int ja, jb;
	int ia, ib;
	fp s;
	int i;
	int j;
	int row;
	int col;
	int ori_row;
	int ori_col;
	int position;
	fp sum;
	int pos_ori;
	fp temp;
	fp temp2;
	int location;
	int cent;
	int tMask_row;
	int tMask_col;
	fp largest_value_current = 0;
	fp largest_value = 0;
	int largest_coordinate_current = 0;
	int largest_coordinate = 0;
	fp fin_max_val = 0;
	int fin_max_coo = 0;
	int largest_row;
	int largest_col;
	int offset_row;
	int offset_col;
	fp mean;
	fp mean_sqr;
	fp variance;
	fp deviation;
	int pointer;
	int ori_pointer;
	int loc_pointer;


	//======================================================================================================================================================150
	//	BLOCK/THREAD IDs
	//======================================================================================================================================================150

  int bx = blockIdx.x;  // get current horizontal block index (0-n)
  int tx = threadIdx.x;	// get current horizontal thread index (0-n)
	int ei_new;

	//======================================================================================================================================================150
	//	UNIQUE STRUCTURE RECONSTRUCTED HERE
	//======================================================================================================================================================150

	// common

	// offsets for either endo or epi points (separate arrays for endo and epi points)
	int d_unique_point_no = bx < d_common.endoPoints ? bx : bx-d_common.endoPoints;

	int* d_unique_d_Row = bx < d_common.endoPoints ? d_endoRow: d_epiRow;
	int* d_unique_d_Col = bx < d_common.endoPoints ? d_endoCol: d_epiCol;
	int* d_unique_d_tRowLoc = bx < d_common.endoPoints ? d_tEndoRowLoc: d_tEpiRowLoc;
	int* d_unique_d_tColLoc = bx < d_common.endoPoints ? d_tEndoColLoc: d_tEpiColLoc;
	fp*  d_in = bx < d_common.endoPoints ? &d_endoT[d_unique_point_no * d_common.in_elem] :
                                               &d_epiT[d_unique_point_no * d_common.in_elem] ;


	// offsets for all points (one array for all points)
	fp*  d_unique_d_in2 = &d_in2[bx*d_common.in2_elem];
	fp*  d_unique_d_conv = &d_conv[bx*d_common.conv_elem];
	fp*  d_unique_d_in2_pad_cumv = &d_in2_pad_cumv[bx*d_common.in2_pad_cumv_elem];
	fp*  d_unique_d_in2_pad_cumv_sel = &d_in2_pad_cumv_sel[bx*d_common.in2_pad_cumv_sel_elem];
	fp*  d_unique_d_in2_sub_cumh = &d_in2_sub_cumh[bx*d_common.in2_sub_cumh_elem];
	fp*  d_unique_d_in2_sub_cumh_sel = &d_in2_sub_cumh_sel[bx*d_common.in2_sub_cumh_sel_elem];
	fp*  d_unique_d_in2_sub2 = &d_in2_sub2[bx*d_common.in2_sub2_elem];
	fp*  d_unique_d_in2_sqr = &d_in2_sqr[bx*d_common.in2_sqr_elem];
	fp*  d_unique_d_in2_sqr_sub2 = &d_in2_sqr_sub2[bx*d_common.in2_sqr_sub2_elem];
	fp*  d_unique_d_in_sqr = &d_in_sqr[bx*d_common.in_sqr_elem];
	fp*  d_unique_d_tMask = &d_tMask[bx*d_common.tMask_elem];
	fp*  d_unique_d_mask_conv = &d_mask_conv[bx*d_common.mask_conv_elem];

	// used to be local
	fp*  d_unique_d_in_mod_temp = &d_in_mod_temp[bx*d_common.in_elem];
	fp*  d_unique_d_in_partial_sum = &d_in_partial_sum[bx*d_common.in_cols];
	fp*  d_unique_d_in_sqr_partial_sum = &d_in_sqr_partial_sum[bx*d_common.in_sqr_rows];
	fp*  d_unique_d_par_max_val = &d_par_max_val[bx*d_common.mask_conv_rows];
	fp*  d_unique_d_par_max_coo = &d_par_max_coo[bx*d_common.mask_conv_rows];

	fp*  d_unique_d_in_final_sum = &d_in_final_sum[bx];
	fp*  d_unique_d_in_sqr_final_sum = &d_in_sqr_final_sum[bx];
	fp*  d_unique_d_denomT = &d_denomT[bx];

	//======================================================================================================================================================150
	//	END
	//======================================================================================================================================================150

	//======================================================================================================================================================150
	//	Initialize checksum
	//======================================================================================================================================================150
#ifdef TEST_CHECKSUM
	if(bx==0 && tx==0){

		for(i=0; i<CHECK; i++){
			checksum[i] = 0;
		}

	}
#endif
	//======================================================================================================================================================150
	//	INITIAL COORDINATE AND TEMPLATE UPDATE
	//======================================================================================================================================================150

	// generate templates based on the first frame only
	if(frame_no == 0){

		//====================================================================================================100
		//	UPDATE ROW LOC AND COL LOC
		//====================================================================================================100

		// uptade temporary endo/epi row/col coordinates (in each block corresponding to point, narrow work to one thread)
		ei_new = tx;
		if(ei_new == 0){

			// update temporary row/col coordinates
			pointer = d_unique_point_no*d_common.no_frames+frame_no;
			d_unique_d_tRowLoc[pointer] = d_unique_d_Row[d_unique_point_no];
			d_unique_d_tColLoc[pointer] = d_unique_d_Col[d_unique_point_no];

		}

		//====================================================================================================100
		//	CREATE TEMPLATES
		//====================================================================================================100

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique_d_Row[d_unique_point_no] - 25 + row - 1;
			ori_col = d_unique_d_Col[d_unique_point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[col*d_common.in_rows+row] = d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		__syncthreads();

		//====================================================================================================100
		//	checksum
		//====================================================================================================100
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[0] = checksum[0]+d_in[i];
			}
		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		__syncthreads();
#endif
		//====================================================================================================100
		//	End
		//====================================================================================================100

	}

	//======================================================================================================================================================150
	//	PROCESS POINTS
	//======================================================================================================================================================150

	// process points in all frames except for the first one
	if(frame_no != 0){

		//====================================================================================================100
		//	Initialize frame-specific variables
		//====================================================================================================100

		//====================================================================================================100
		//	SELECTION
		//====================================================================================================100

		in2_rowlow = d_unique_d_Row[d_unique_point_no] - d_common.sSize;													// (1 to n+1)
		in2_collow = d_unique_d_Col[d_unique_point_no] - d_common.sSize;

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_rows == 0){
				row = d_common.in2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + in2_rowlow - 1;
			ori_col = col + in2_collow - 1;
			d_unique_d_in2[ei_new] = d_frame[ori_col*d_common.frame_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		__syncthreads();

		//====================================================================================================100
		//	checksum
		//====================================================================================================100
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_elem; i++){
				checksum[1] = checksum[1]+d_unique_d_in2[i];
			}
		}

		//====================================================================================================100
		//	SYNCHRONIZE THREADS
		//====================================================================================================100

		__syncthreads();
#endif
		//====================================================================================================100
		//	CONVOLUTION
		//====================================================================================================100

		//==================================================50
		//	ROTATION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){
		// while(ei_new < 1){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// execution
			rot_row = (d_common.in_rows-1) - row;
			rot_col = (d_common.in_rows-1) - col;
			d_unique_d_in_mod_temp[ei_new] = d_in[rot_col*d_common.in_rows+rot_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[2] = checksum[2]+d_unique_d_in_mod_temp[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	ACTUAL CONVOLUTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.conv_elem){

			// figure out row/col location in array
			ic = (ei_new+1) % d_common.conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.conv_rows == 0){
				ic = d_common.conv_rows;
				jc = jc-1;
			}

			//
			j = jc + d_common.joffset;
			jp1 = j + 1;
			if(d_common.in2_cols < jp1){
				ja1 = jp1 - d_common.in2_cols;
			}
			else{
				ja1 = 1;
			}
			if(d_common.in_cols < j){
				ja2 = d_common.in_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + d_common.ioffset;
			ip1 = i + 1;

			if(d_common.in2_rows < ip1){
				ia1 = ip1 - d_common.in2_rows;
			}
			else{
				ia1 = 1;
			}
			if(d_common.in_rows < i){
				ia2 = d_common.in_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_unique_d_in_mod_temp[d_common.in_rows*(ja-1)+ia-1] * d_unique_d_in2[d_common.in2_rows*(jb-1)+ib-1];
				}
			}

			//d_unique_d_conv[d_common.conv_rows*(jc-1)+ic-1] = s;
			d_unique_d_conv[ei_new] = s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.conv_elem; i++){
				checksum[3] = checksum[3]+d_unique_d_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		// 	CUMULATIVE SUM	(LOCAL)
		//====================================================================================================100

		//==================================================50
		//	PADD ARRAY
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_elem){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&														// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_rows) &&
				col > (d_common.in2_pad_add_cols-1) &&
				col < (d_common.in2_pad_add_cols+d_common.in2_cols)){
				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2[ori_col*d_common.in2_rows+ori_row];
			}
			else{																			// do if otherwise
				d_unique_d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[4] = checksum[4]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	VERTICAL CUMULATIVE SUM
		//==================================================50

		//work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_cols){

			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows; position = position + 1){
				d_unique_d_in2_pad_cumv[position] = d_unique_d_in2_pad_cumv[position] + sum;
				sum = d_unique_d_in2_pad_cumv[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_cols; i++){
				checksum[5] = checksum[5]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				checksum[6] = checksum[6]+d_unique_d_in2_pad_cumv_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[7] = checksum[7]+d_unique_d_in2_sub_cumh[i];
			}
		}
		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif

		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// subtract
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[8] = checksum[8]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_rows){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem; position = position + d_common.in2_sub_cumh_rows){
				d_unique_d_in2_sub_cumh[position] = d_unique_d_in2_sub_cumh[position] + sum;
				sum = d_unique_d_in2_sub_cumh[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[9] = checksum[9]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				checksum[10] = checksum[10]+d_unique_d_in2_sub_cumh_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique_d_in2_sub2[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[11] = checksum[11]+d_unique_d_in2_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// subtract
			d_unique_d_in2_sub2[ei_new] = d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[12] = checksum[12]+d_unique_d_in2_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	CUMULATIVE SUM 2
		//====================================================================================================100

		//==================================================50
		//	MULTIPLICATION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sqr_elem){

			temp = d_unique_d_in2[ei_new];
			d_unique_d_in2_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sqr_elem; i++){
				checksum[13] = checksum[13]+d_unique_d_in2_sqr[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	PAD ARRAY, VERTICAL CUMULATIVE SUM
		//==================================================50

		//==================================================50
		//	PAD ARRAY
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_elem){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&													// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_sqr_rows) &&
				col > (d_common.in2_pad_add_cols-1) &&
				col < (d_common.in2_pad_add_cols+d_common.in2_sqr_cols)){
				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2_sqr[ori_col*d_common.in2_sqr_rows+ori_row];
			}
			else{																							// do if otherwise
				d_unique_d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[14] = checksum[14]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	VERTICAL CUMULATIVE SUM
		//==================================================50

		//work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_cols){

			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows; position = position + 1){
				d_unique_d_in2_pad_cumv[position] = d_unique_d_in2_pad_cumv[position] + sum;
				sum = d_unique_d_in2_pad_cumv[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_elem; i++){
				checksum[15] = checksum[15]+d_unique_d_in2_pad_cumv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_pad_cumv_sel_elem; i++){
				checksum[16] = checksum[16]+d_unique_d_in2_pad_cumv_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[17] = checksum[17]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// subtract
			d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_elem; i++){
				checksum[18] = checksum[18]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_rows){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem; position = position + d_common.in2_sub_cumh_rows){
				d_unique_d_in2_sub_cumh[position] = d_unique_d_in2_sub_cumh[position] + sum;
				sum = d_unique_d_in2_sub_cumh[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_rows; i++){
				checksum[19] = checksum[19]+d_unique_d_in2_sub_cumh[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub_cumh_sel_elem; i++){
				checksum[20] = checksum[20]+d_unique_d_in2_sub_cumh_sel[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SELECTION 2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[21] = checksum[21]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	SUBTRACTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// subtract
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[22] = checksum[22]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	FINAL
		//====================================================================================================100

		//==================================================50
		//	DENOMINATOR A		SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			temp = d_unique_d_in2_sub2[ei_new];
			temp2 = d_unique_d_in2_sqr_sub2[ei_new] - (temp * temp / d_common.in_elem);
			if(temp2 < 0){
				temp2 = 0;
			}
			d_unique_d_in2_sqr_sub2[ei_new] = sqrt(temp2);


			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[23] = checksum[23]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	MULTIPLICATION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in_sqr_elem){

			temp = d_in[ei_new];
			d_unique_d_in_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_sqr_elem; i++){
				checksum[24] = checksum[24]+d_unique_d_in_sqr[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	IN SUM
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in_cols){

			sum = 0;
			for(i = 0; i < d_common.in_rows; i++){

				sum = sum + d_in[ei_new*d_common.in_rows+i];

			}
			d_unique_d_in_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_cols; i++){
				checksum[25] = checksum[25]+d_unique_d_in_partial_sum[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	IN_SQR SUM
		//==================================================50

		ei_new = tx;
		while(ei_new < d_common.in_sqr_rows){

			sum = 0;
			for(i = 0; i < d_common.in_sqr_cols; i++){

				sum = sum + d_unique_d_in_sqr[ei_new+d_common.in_sqr_rows*i];

			}
			d_unique_d_in_sqr_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_sqr_rows; i++){
				checksum[26] = checksum[26]+d_unique_d_in_sqr_partial_sum[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	FINAL SUMMATION
		//==================================================50

		if(tx == 0){

			d_unique_d_in_final_sum[0] = 0;
			for(i = 0; i<d_common.in_cols; i++){
				// in_final_sum = in_final_sum + d_unique_d_in_partial_sum[i];
				d_unique_d_in_final_sum[0] = d_unique_d_in_final_sum[0] + d_unique_d_in_partial_sum[i];
			}

		}else if(tx == 1){

			d_unique_d_in_sqr_final_sum[0] = 0;
			for(i = 0; i<d_common.in_sqr_cols; i++){
				// in_sqr_final_sum = in_sqr_final_sum + d_unique_d_in_sqr_partial_sum[i];
				d_unique_d_in_sqr_final_sum[0] = d_unique_d_in_sqr_final_sum[0] + d_unique_d_in_sqr_partial_sum[i];
			}

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[27] = checksum[27]+d_unique_d_in_final_sum[0]+d_unique_d_in_sqr_final_sum[0];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	DENOMINATOR T
		//==================================================50

		if(tx == 0){

			// mean = in_final_sum / d_common.in_elem;													// gets mean (average) value of element in ROI
			mean = d_unique_d_in_final_sum[0] / d_common.in_elem;													// gets mean (average) value of element in ROI
			mean_sqr = mean * mean;
			// variance  = (in_sqr_final_sum / d_common.in_elem) - mean_sqr;							// gets variance of ROI
			variance  = (d_unique_d_in_sqr_final_sum[0] / d_common.in_elem) - mean_sqr;							// gets variance of ROI
			deviation = sqrt(variance);																// gets standard deviation of ROI

			// denomT = sqrt((float)(d_common.in_elem-1))*deviation;
			d_unique_d_denomT[0] = sqrt((float)(d_common.in_elem-1))*deviation;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[28] = checksum[28]+d_unique_d_denomT[i];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * denomT;
			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * d_unique_d_denomT[0];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[29] = checksum[29]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	NUMERATOR	SAVE RESULT IN CONVOLUTION
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.conv_elem){

			// d_unique_d_conv[ei_new] = d_unique_d_conv[ei_new] - d_unique_d_in2_sub2[ei_new] * in_final_sum / d_common.in_elem;
			d_unique_d_conv[ei_new] = d_unique_d_conv[ei_new] - d_unique_d_in2_sub2[ei_new] * d_unique_d_in_final_sum[0] / d_common.in_elem;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.conv_elem; i++){
				checksum[30] = checksum[30]+d_unique_d_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	CORRELATION	SAVE RESULT IN CUMULATIVE SUM A2
		//==================================================50

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_conv[ei_new] / d_unique_d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}



		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in2_sub2_elem; i++){
				checksum[31] = checksum[31]+d_unique_d_in2_sqr_sub2[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	TEMPLATE MASK CREATE
		//====================================================================================================100

		cent = d_common.sSize + d_common.tSize + 1;
		if(frame_no == 0){
			tMask_row = cent + d_unique_d_Row[d_unique_point_no] - d_unique_d_Row[d_unique_point_no] - 1;
			tMask_col = cent + d_unique_d_Col[d_unique_point_no] - d_unique_d_Col[d_unique_point_no] - 1;
		}
		else{
			pointer = d_unique_point_no*d_common.no_frames+frame_no-1;
			tMask_row = cent + d_unique_d_tRowLoc[pointer] - d_unique_d_Row[d_unique_point_no] - 1;
			tMask_col = cent + d_unique_d_tColLoc[pointer] - d_unique_d_Col[d_unique_point_no] - 1;
		}

		//work
		ei_new = tx;
		while(ei_new < d_common.tMask_elem){

			location = tMask_col*d_common.tMask_rows + tMask_row;

			if(ei_new==location){
				d_unique_d_tMask[ei_new] = 1;
			}
			else{
				d_unique_d_tMask[ei_new] = 0;
			}

			//go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.tMask_elem; i++){
				checksum[32] = checksum[32]+d_unique_d_tMask[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	MASK CONVOLUTION
		//====================================================================================================100

		// work
		ei_new = tx;
		while(ei_new < d_common.mask_conv_elem){

			// figure out row/col location in array
			ic = (ei_new+1) % d_common.mask_conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.mask_conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.mask_conv_rows == 0){
				ic = d_common.mask_conv_rows;
				jc = jc-1;
			}

			//
			j = jc + d_common.mask_conv_joffset;
			jp1 = j + 1;
			if(d_common.mask_cols < jp1){
				ja1 = jp1 - d_common.mask_cols;
			}
			else{
				ja1 = 1;
			}
			if(d_common.tMask_cols < j){
				ja2 = d_common.tMask_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + d_common.mask_conv_ioffset;
			ip1 = i + 1;

			if(d_common.mask_rows < ip1){
				ia1 = ip1 - d_common.mask_rows;
			}
			else{
				ia1 = 1;
			}
			if(d_common.tMask_rows < i){
				ia2 = d_common.tMask_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_unique_d_tMask[d_common.tMask_rows*(ja-1)+ia-1] * 1;
				}
			}

			// //d_unique_d_mask_conv[d_common.mask_conv_rows*(jc-1)+ic-1] = s;
			d_unique_d_mask_conv[ei_new] = d_unique_d_in2_sqr_sub2[ei_new] * s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.mask_conv_elem; i++){
				checksum[33] = checksum[33]+d_unique_d_mask_conv[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	MAXIMUM VALUE
		//====================================================================================================100

		//==================================================50
		//	INITIAL SEARCH
		//==================================================50

		ei_new = tx;
		while(ei_new < d_common.mask_conv_rows){

			for(i=0; i<d_common.mask_conv_cols; i++){
				largest_coordinate_current = ei_new*d_common.mask_conv_rows+i;
				largest_value_current = fabs(d_unique_d_mask_conv[largest_coordinate_current]);
				if(largest_value_current > largest_value){
					largest_coordinate = largest_coordinate_current;
					largest_value = largest_value_current;
				}
			}
			d_unique_d_par_max_coo[ei_new] = largest_coordinate;
			d_unique_d_par_max_val[ei_new] = largest_value;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.mask_conv_rows; i++){
				checksum[34] = checksum[34]+d_unique_d_par_max_coo[i]+d_unique_d_par_max_val[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	FINAL SEARCH
		//==================================================50

		if(tx == 0){

			for(i = 0; i < d_common.mask_conv_rows; i++){
				if(d_unique_d_par_max_val[i] > fin_max_val){
					fin_max_val = d_unique_d_par_max_val[i];
					fin_max_coo = d_unique_d_par_max_coo[i];
				}
			}

			// convert coordinate to row/col form
			largest_row = (fin_max_coo+1) % d_common.mask_conv_rows - 1;											// (0-n) row
			largest_col = (fin_max_coo+1) / d_common.mask_conv_rows;												// (0-n) column
			if((fin_max_coo+1) % d_common.mask_conv_rows == 0){
				largest_row = d_common.mask_conv_rows - 1;
				largest_col = largest_col - 1;
			}

			// calculate offset
			largest_row = largest_row + 1;																	// compensate to match MATLAB format (1-n)
			largest_col = largest_col + 1;																	// compensate to match MATLAB format (1-n)
			offset_row = largest_row - d_common.in_rows - (d_common.sSize - d_common.tSize);
			offset_col = largest_col - d_common.in_cols - (d_common.sSize - d_common.tSize);
			pointer = d_unique_point_no*d_common.no_frames+frame_no;
			d_unique_d_tRowLoc[pointer] = d_unique_d_Row[d_unique_point_no] + offset_row;
			d_unique_d_tColLoc[pointer] = d_unique_d_Col[d_unique_point_no] + offset_col;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			checksum[35] = checksum[35]+d_unique_d_tRowLoc[pointer]+d_unique_d_tColLoc[pointer];
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

		//====================================================================================================100
		//	End
		//====================================================================================================100

	}

	//======================================================================================================================================================150
	//	PERIODIC COORDINATE AND TEMPLATE UPDATE
	//======================================================================================================================================================150

	if(frame_no != 0 && (frame_no)%10 == 0){


		//====================================================================================================100
		// if the last frame in the bath, update template
		//====================================================================================================100

		// update coordinate
		loc_pointer = d_unique_point_no*d_common.no_frames+frame_no;

		d_unique_d_Row[d_unique_point_no] = d_unique_d_tRowLoc[loc_pointer];
		d_unique_d_Col[d_unique_point_no] = d_unique_d_tColLoc[loc_pointer];

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique_d_Row[d_unique_point_no] - 25 + row - 1;
			ori_col = d_unique_d_Col[d_unique_point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[ei_new] = d_common.alpha*d_in[ei_new] + (1-d_common.alpha)*d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();

		//==================================================50
		//	checksum
		//==================================================50
#ifdef TEST_CHECKSUM
		if(bx==0 && tx==0){
			for(i=0; i<d_common.in_elem; i++){
				checksum[36] = checksum[36]+d_in[i];
			}
		}

		//==================================================50
		//	SYNCHRONIZE THREADS
		//==================================================50

		__syncthreads();
#endif
		//==================================================50
		//	End
		//==================================================50

	}
}

//========================================================================================================================================================================================================200
//  INLINED: kernel/kernel.cu  (kernel_gpu_wrapper function body)
//========================================================================================================================================================================================================200

uint64_t
kernel_gpu_wrapper(  params_common common,
    int* endoRow,
    int* endoCol,
    int* tEndoRowLoc,
    int* tEndoColLoc,
    int* epiRow,
    int* epiCol,
    int* tEpiRowLoc,
    int* tEpiColLoc,
    avi_t* frames)
{

  // common
  common.in_rows = common.tSize + 1 + common.tSize;
  common.in_cols = common.in_rows;
  common.in_elem = common.in_rows * common.in_cols;
  common.in_mem = sizeof(fp) * common.in_elem;

  //==================================================50
  // endo points templates
  //==================================================50

  fp* d_endoT;
  cudaMalloc((void**)&d_endoT, common.in_mem * common.endoPoints);

  //==================================================50
  // epi points templates
  //==================================================50

  fp* d_epiT;
  cudaMalloc((void**)&d_epiT, common.in_mem * common.epiPoints);

  //====================================================================================================100
  //   AREA AROUND POINT    FROM  FRAME  (LOCAL)
  //====================================================================================================100

  common.in2_rows = common.sSize + 1 + common.sSize;
  common.in2_cols = common.in2_rows;
  common.in2_elem = common.in2_rows * common.in2_cols;
  common.in2_mem = sizeof(fp) * common.in2_elem;

  fp* d_in2;
  cudaMalloc((void**)&d_in2, common.in2_mem * common.allPoints);

  //====================================================================================================100
  //   CONVOLUTION  (LOCAL)
  //====================================================================================================100

  common.conv_rows = common.in_rows + common.in2_rows - 1;
  common.conv_cols = common.in_cols + common.in2_cols - 1;
  common.conv_elem = common.conv_rows * common.conv_cols;
  common.conv_mem = sizeof(fp) * common.conv_elem;
  common.ioffset = 0;
  common.joffset = 0;

  fp* d_conv;
  cudaMalloc((void**)&d_conv, common.conv_mem * common.allPoints);

  //====================================================================================================100
  //   CUMULATIVE SUM  (LOCAL)
  //====================================================================================================100

  common.in2_pad_add_rows = common.in_rows;
  common.in2_pad_add_cols = common.in_cols;

  common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
  common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
  common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
  common.in2_pad_cumv_mem = sizeof(fp) * common.in2_pad_cumv_elem;

  fp* d_in2_pad_cumv;
  cudaMalloc((void**)&d_in2_pad_cumv, common.in2_pad_cumv_mem * common.allPoints);

  common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;
  common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
  common.in2_pad_cumv_sel_collow = 1;
  common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
  common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
  common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
  common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
  common.in2_pad_cumv_sel_mem = sizeof(fp) * common.in2_pad_cumv_sel_elem;

  fp* d_in2_pad_cumv_sel;
  cudaMalloc((void**)&d_in2_pad_cumv_sel, common.in2_pad_cumv_sel_mem * common.allPoints);

  common.in2_pad_cumv_sel2_rowlow = 1;
  common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
  common.in2_pad_cumv_sel2_collow = 1;
  common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
  common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
  common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
  common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
  common.in2_sub_cumh_mem = sizeof(fp) * common.in2_sub_cumh_elem;

  fp* d_in2_sub_cumh;
  cudaMalloc((void**)&d_in2_sub_cumh, common.in2_sub_cumh_mem * common.allPoints);

  common.in2_sub_cumh_sel_rowlow = 1;
  common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
  common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
  common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
  common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
  common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
  common.in2_sub_cumh_sel_mem = sizeof(fp) * common.in2_sub_cumh_sel_elem;

  fp* d_in2_sub_cumh_sel;
  cudaMalloc((void**)&d_in2_sub_cumh_sel, common.in2_sub_cumh_sel_mem * common.allPoints);

  common.in2_sub_cumh_sel2_rowlow = 1;
  common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel2_collow = 1;
  common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
  common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
  common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
  common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
  common.in2_sub2_mem = sizeof(fp) * common.in2_sub2_elem;

  fp* d_in2_sub2;
  cudaMalloc((void**)&d_in2_sub2, common.in2_sub2_mem * common.allPoints);

  //====================================================================================================100
  //  CUMULATIVE SUM 2  (LOCAL)
  //====================================================================================================100

  common.in2_sqr_rows = common.in2_rows;
  common.in2_sqr_cols = common.in2_cols;
  common.in2_sqr_elem = common.in2_elem;
  common.in2_sqr_mem = common.in2_mem;

  fp* d_in2_sqr;
  cudaMalloc((void**)&d_in2_sqr, common.in2_sqr_mem * common.allPoints);

  common.in2_sqr_sub2_rows = common.in2_sub2_rows;
  common.in2_sqr_sub2_cols = common.in2_sub2_cols;
  common.in2_sqr_sub2_elem = common.in2_sub2_elem;
  common.in2_sqr_sub2_mem = common.in2_sub2_mem;

  fp* d_in2_sqr_sub2;
  cudaMalloc((void**)&d_in2_sqr_sub2, common.in2_sqr_sub2_mem * common.allPoints);

  //====================================================================================================100
  //  FINAL  (LOCAL)
  //====================================================================================================100

  common.in_sqr_rows = common.in_rows;
  common.in_sqr_cols = common.in_cols;
  common.in_sqr_elem = common.in_elem;
  common.in_sqr_mem = common.in_mem;

  fp* d_in_sqr;
  cudaMalloc((void**)&d_in_sqr, common.in_sqr_mem * common.allPoints);

  //====================================================================================================100
  //  TEMPLATE MASK CREATE  (LOCAL)
  //====================================================================================================100

  common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
  common.tMask_cols = common.tMask_rows;
  common.tMask_elem = common.tMask_rows * common.tMask_cols;
  common.tMask_mem = sizeof(fp) * common.tMask_elem;

  fp* d_tMask;
  cudaMalloc((void**)&d_tMask, common.tMask_mem * common.allPoints);

  //====================================================================================================100
  //  POINT MASK INITIALIZE  (LOCAL)
  //====================================================================================================100

  common.mask_rows = common.maxMove;
  common.mask_cols = common.mask_rows;
  common.mask_elem = common.mask_rows * common.mask_cols;
  common.mask_mem = sizeof(fp) * common.mask_elem;

  //====================================================================================================100
  //  MASK CONVOLUTION  (LOCAL)
  //====================================================================================================100

  common.mask_conv_rows = common.tMask_rows;
  common.mask_conv_cols = common.tMask_cols;
  common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;
  common.mask_conv_mem = sizeof(fp) * common.mask_conv_elem;
  common.mask_conv_ioffset = (common.mask_rows-1)/2;
  if((common.mask_rows-1) % 2 > 0.5){
    common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
  }
  common.mask_conv_joffset = (common.mask_cols-1)/2;
  if((common.mask_cols-1) % 2 > 0.5){
    common.mask_conv_joffset = common.mask_conv_joffset + 1;
  }

  int* d_endoRow;
  cudaMalloc((void**)&d_endoRow, common.endo_mem);
  cudaMemcpy(d_endoRow, endoRow, common.endo_mem, cudaMemcpyHostToDevice);

  int* d_endoCol;
  cudaMalloc((void**)&d_endoCol, common.endo_mem);
  cudaMemcpy(d_endoCol, endoCol, common.endo_mem, cudaMemcpyHostToDevice);

  int* d_tEndoRowLoc;
  int* d_tEndoColLoc;
  cudaMalloc((void**)&d_tEndoRowLoc, common.endo_mem*common.no_frames);
  cudaMemcpy(d_tEndoRowLoc, tEndoRowLoc, common.endo_mem*common.no_frames, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_tEndoColLoc, common.endo_mem*common.no_frames);
  cudaMemcpy(d_tEndoColLoc, tEndoColLoc, common.endo_mem*common.no_frames, cudaMemcpyHostToDevice);

  int* d_epiRow;
  int* d_epiCol;
  cudaMalloc((void**)&d_epiRow, common.epi_mem);
  cudaMemcpy(d_epiRow, epiRow, common.epi_mem, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_epiCol, common.epi_mem);
  cudaMemcpy(d_epiCol, epiCol, common.epi_mem, cudaMemcpyHostToDevice);

  int* d_tEpiRowLoc;
  int* d_tEpiColLoc;
  cudaMalloc((void**)&d_tEpiRowLoc, common.epi_mem*common.no_frames);
  cudaMemcpy(d_tEpiRowLoc, tEpiRowLoc, common.epi_mem*common.no_frames, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_tEpiColLoc, common.epi_mem*common.no_frames);
  cudaMemcpy(d_tEpiColLoc, tEpiColLoc, common.epi_mem*common.no_frames, cudaMemcpyHostToDevice);

  fp* d_mask_conv;
  cudaMalloc((void**)&d_mask_conv, common.mask_conv_mem * common.allPoints);

  fp* d_in_mod_temp;
  cudaMalloc((void**)&d_in_mod_temp, common.in_mem * common.allPoints);

  fp* d_in_partial_sum;
  cudaMalloc((void**)&d_in_partial_sum, sizeof(fp)*common.in_cols * common.allPoints);

  fp* d_in_sqr_partial_sum;
  cudaMalloc((void**)&d_in_sqr_partial_sum, sizeof(fp)*common.in_sqr_rows * common.allPoints);

  fp* d_par_max_val;
  cudaMalloc((void**)&d_par_max_val, sizeof(fp)*common.mask_conv_rows * common.allPoints);

  fp* d_par_max_coo;
  cudaMalloc((void**)&d_par_max_coo, sizeof(fp)*common.mask_conv_rows * common.allPoints);

  fp* d_in_final_sum;
  cudaMalloc((void**)&d_in_final_sum, sizeof(fp)*common.allPoints);

  fp* d_in_sqr_final_sum;
  cudaMalloc((void**)&d_in_sqr_final_sum, sizeof(fp)*common.allPoints);

  fp* d_denomT;
  cudaMalloc((void**)&d_denomT, sizeof(fp)*common.allPoints);

#ifdef TEST_CHECKSUM
  fp* checksum = (fp*) malloc (sizeof(fp)*CHECK);
  fp* d_checksum;
  cudaMalloc((void**)&d_checksum, sizeof(fp)*CHECK);
#endif

  //====================================================================================================100
  //  EXECUTION PARAMETERS
  //====================================================================================================100

  dim3 threads(NUMBER_THREADS);
  dim3 grids(common.allPoints);

  printf("frame progress: ");
  fflush(NULL);

  //====================================================================================================100
  //  LAUNCH
  //====================================================================================================100

  fp* frame;
  int frame_no;

  fp* d_frame;
  cudaMalloc((void**)&d_frame, sizeof(fp)*common.frame_elem);

  uint64_t kernel_time = 0;

  for(frame_no=0; frame_no<common.frames_processed; frame_no++) {

    frame = get_frame(  frames,
        frame_no,
        0,
        0,
        1);

    cudaMemcpy(d_frame, frame, sizeof(fp)*common.frame_elem, cudaMemcpyHostToDevice);

    uint64_t start_time = get_time();

    hw<<<grids, threads>>>(
        frame_no,
        common,
        d_frame,
        d_endoRow,
        d_endoCol,
        d_tEndoRowLoc,
        d_tEndoColLoc,
        d_epiRow,
        d_epiCol,
        d_tEpiRowLoc,
        d_tEpiColLoc,
        d_endoT,
        d_epiT,
        d_in2,
        d_conv,
        d_in2_pad_cumv,
        d_in2_pad_cumv_sel,
        d_in2_sub_cumh,
        d_in2_sub_cumh_sel,
        d_in2_sub2,
        d_in2_sqr,
        d_in2_sqr_sub2,
        d_in_sqr,
        d_tMask,
        d_mask_conv,
        d_in_mod_temp,
        d_in_partial_sum,
        d_in_sqr_partial_sum,
        d_par_max_val,
        d_par_max_coo,
        d_in_final_sum,
        d_in_sqr_final_sum,
        d_denomT
#ifdef TEST_CHECKSUM
          ,d_checksum
#endif
          );

    cudaDeviceSynchronize();
    uint64_t end_time = get_time();
    kernel_time += end_time - start_time;

    free(frame);

    printf("%d ", frame_no);
    fflush(NULL);

#ifdef TEST_CHECKSUM
    cudaMemcpy(checksum, d_checksum, sizeof(fp)*CHECK, cudaMemcpyDeviceToHost);
    printf("CHECKSUM:\n");
    for(int i=0; i<CHECK; i++){
      printf("i=%d checksum=%f\n", i, checksum[i]);
    }
    printf("\n\n");
#endif

  }
  uint64_t end_time = get_time();

  cudaMemcpy(tEndoRowLoc, d_tEndoRowLoc, common.endo_mem * common.no_frames, cudaMemcpyDeviceToHost);
  cudaMemcpy(tEndoColLoc, d_tEndoColLoc, common.endo_mem * common.no_frames, cudaMemcpyDeviceToHost);
  cudaMemcpy(tEpiRowLoc, d_tEpiRowLoc, common.epi_mem * common.no_frames, cudaMemcpyDeviceToHost);
  cudaMemcpy(tEpiColLoc, d_tEpiColLoc, common.epi_mem * common.no_frames, cudaMemcpyDeviceToHost);

#ifdef TEST_CHECKSUM
  free(checksum);
  cudaFree(d_checksum);
#endif
  cudaFree(d_epiT);
  cudaFree(d_endoT);
  cudaFree(d_in2);
  cudaFree(d_conv);
  cudaFree(d_in2_pad_cumv);
  cudaFree(d_in2_pad_cumv_sel);
  cudaFree(d_in2_sub_cumh);
  cudaFree(d_in2_sub_cumh_sel);
  cudaFree(d_in2_sub2);
  cudaFree(d_in2_sqr);
  cudaFree(d_in2_sqr_sub2);
  cudaFree(d_in_sqr);
  cudaFree(d_tMask);
  cudaFree(d_endoRow);
  cudaFree(d_endoCol);
  cudaFree(d_tEndoRowLoc);
  cudaFree(d_tEndoColLoc);
  cudaFree(d_epiRow);
  cudaFree(d_epiCol);
  cudaFree(d_tEpiRowLoc);
  cudaFree(d_tEpiColLoc);
  cudaFree(d_mask_conv);
  cudaFree(d_in_mod_temp);
  cudaFree(d_in_partial_sum);
  cudaFree(d_in_sqr_partial_sum);
  cudaFree(d_par_max_val);
  cudaFree(d_par_max_coo);
  cudaFree(d_in_final_sum);
  cudaFree(d_in_sqr_final_sum);
  cudaFree(d_denomT);
  cudaFree(d_frame);

  printf("\n");
  fflush(NULL);
  return kernel_time;
}

//========================================================================================================================================================================================================200
//  INLINED: main.cu  (main() function)
//========================================================================================================================================================================================================200

int main(int argc, char* argv []){

  printf("Workgroup size of kernel = %d \n", NUMBER_THREADS);

  //======================================================================================================================================================150
  //  VARIABLES
  //======================================================================================================================================================150

  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;

  avi_t* frames;

  time0 = get_time();

  //======================================================================================================================================================150
  //  STRUCTURES, GLOBAL STRUCTURE VARIABLES
  //======================================================================================================================================================150

  params_common common;
  common.common_mem = sizeof(params_common);

  //======================================================================================================================================================150
  //   FRAME INFO
  //======================================================================================================================================================150

  char* video_file_name;

  video_file_name = (char *) "../data/heartwall/test.avi";
  frames = (avi_t*)AVI_open_input_file(video_file_name, 1);
  if (frames == NULL)  {
    AVI_print_error((char *) "Error with AVI_open_input_file");
    return -1;
  }

  common.no_frames = AVI_video_frames(frames);
  common.frame_rows = AVI_video_height(frames);
  common.frame_cols = AVI_video_width(frames);
  common.frame_elem = common.frame_rows * common.frame_cols;
  common.frame_mem = sizeof(fp) * common.frame_elem;

  time1 = get_time();

  //======================================================================================================================================================150
  //   CHECK INPUT ARGUMENTS
  //======================================================================================================================================================150

  if(argc!=2){
    printf("ERROR: missing argument (number of frames to process) or too many arguments\n");
    return 0;
  }
  else{
    common.frames_processed = atoi(argv[1]);
    if(common.frames_processed<0 || common.frames_processed>common.no_frames){
      printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n",
             common.frames_processed, common.no_frames);
      return 0;
    }
  }

  time2 = get_time();

  //======================================================================================================================================================150
  //  INPUTS
  //======================================================================================================================================================150

  char* param_file_name = (char *) "../data/heartwall/input.txt";
  read_parameters(param_file_name,
                  &common.tSize,
                  &common.sSize,
                  &common.maxMove,
                  &common.alpha);

  read_header(param_file_name, &common.endoPoints, &common.epiPoints);

  common.allPoints = common.endoPoints + common.epiPoints;

  common.endo_mem = sizeof(int) * common.endoPoints;

  int* endoRow;
  endoRow = (int*)malloc(common.endo_mem);
  int* endoCol;
  endoCol = (int*)malloc(common.endo_mem);
  int* tEndoRowLoc;
  tEndoRowLoc = (int*)malloc(common.endo_mem * common.no_frames);
  int* tEndoColLoc;
  tEndoColLoc = (int*)malloc(common.endo_mem * common.no_frames);

  common.epi_mem = sizeof(int) * common.epiPoints;

  int* epiRow;
  epiRow = (int *)malloc(common.epi_mem);
  int* epiCol;
  epiCol = (int *)malloc(common.epi_mem);
  int* tEpiRowLoc;
  tEpiRowLoc = (int *)malloc(common.epi_mem * common.no_frames);
  int* tEpiColLoc;
  tEpiColLoc = (int *)malloc(common.epi_mem * common.no_frames);

  read_data(param_file_name,
            common.endoPoints,
            endoRow,
            endoCol,
            common.epiPoints,
            epiRow,
            epiCol);

  time3 = get_time();

  //======================================================================================================================================================150
  //  KERNELL WRAPPER CALL
  //======================================================================================================================================================150

  uint64_t kernelTime = kernel_gpu_wrapper(common,
                     endoRow,
                     endoCol,
                     tEndoRowLoc,
                     tEndoColLoc,
                     epiRow,
                     epiCol,
                     tEpiRowLoc,
                     tEpiColLoc,
                     frames);

  time4 = get_time();

#ifdef OUTPUT
  write_data("result.txt",
             common.no_frames,
             common.frames_processed,
             common.endoPoints,
             tEndoRowLoc,
             tEndoColLoc,
             common.epiPoints,
             tEpiRowLoc,
             tEpiColLoc);
#endif

  //======================================================================================================================================================150
  //  DEALLOCATION
  //======================================================================================================================================================150

  free(endoRow);
  free(endoCol);
  free(tEndoRowLoc);
  free(tEndoColLoc);

  free(epiRow);
  free(epiCol);
  free(tEpiRowLoc);
  free(tEpiColLoc);

  time5= get_time();

  //======================================================================================================================================================150
  //  DISPLAY TIMING
  //======================================================================================================================================================150

  printf("Time spent in different stages of the application:\n");
  printf("%15.12f s, %15.12f : READ INITIAL VIDEO FRAME\n",
      (fp) (time1-time0) / 1000000, (fp) (time1-time0) / (fp) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : READ COMMAND LINE PARAMETERS\n",
      (fp) (time2-time1) / 1000000, (fp) (time2-time1) / (fp) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : READ INPUTS FROM FILE\n",
      (fp) (time3-time2) / 1000000, (fp) (time3-time2) / (fp) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : GPU ALLOCATION, COPYING, COMPUTATION\n",
      (fp) (time4-time3) / 1000000, (fp) (time4-time3) / (fp) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : GPU KERNELS\n",
      (fp) (kernelTime) / 1000000, (fp) (kernelTime) / (fp) (time5-time0) * 100);
  printf("%15.12f s, %15.12f : FREE MEMORY\n",
      (fp) (time5-time4) / 1000000, (fp) (time5-time4) / (fp) (time5-time0) * 100);
  printf("Total time:\n");
  printf("%15.12f s\n", (fp) (time5-time0) / 1000000);

}
