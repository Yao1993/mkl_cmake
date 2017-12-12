#include <yao/io/binary_io.h>
#include <vector>
#include <mkl.h>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <yao/time/timer.h>

void Bench()
{


	std::vector<float> raw_data;
	yao::io::ReadBinaryArray("C:/Users/tm42h7/Desktop/TPS/DoseMatrix/demo.bin", raw_data);
	int num_voxels = 69401;
	int num_spots = 6987;

	float* row_major = raw_data.data();

	float alpha = 1.0f; 
	char ordering = 'r';
	char trans = 't';
	int m = num_voxels;
	int n = num_spots;
	int lda = n;
	int ldb = m;


	MKL_INT job[6];
	job[0] = 0;    // the rectangular matrix A is converted to the CSR format;
	job[1] = 0;    // zero-based indexing for the rectangular matrix A is used;
	job[2] = 0;    // zero-based indexing for the matrix in CSR format is used;
	job[3] = 2;    // adns is a whole matrix A;
	job[4] = 1e9;  // maximum number of the non-zero elements allowed if job[0]=0;
	job[5] = 0;    //If job[5]=0, only array ia is generated for the output storage.
	               //If job[5]>0, arrays acsr, ia, ja are generated for the output storage.

	m = num_voxels;
	n = num_spots;
	int *ia = new int[m + 1];
	MKL_INT info;
	yao::time::Timer timer(true);
	mkl_sdnscsr(job, &m, &n, row_major, &n, NULL, NULL, ia, &info);
	std::cout << "Elapsed time of counting nnz per row :" << timer << " ms" << std::endl;
	std::cout << "mkl_sdnscsr: " << info << std::endl;

	int nnz = ia[m] - ia[0];
	float *csr = new float[nnz];
	int *ja = new int[nnz];
	job[5] = 1;
	timer.Reset();
	mkl_sdnscsr(job, &m, &n, row_major, &n, csr, ja, ia, &info);
	std::cout << "Convert dense matrix to csr matrix :" << timer << " ms" << std::endl;
	std::cout << "mkl_sdnscsr: " << info << std::endl;


	float *x = new float[n];
	std::fill_n(x, n, 1.0f);
	float *y = new float[m];
	std::fill_n(y, m, 0.0f);
	trans = 'n';
	char matdescra[6] = { 'G', 'I', 'I', 'C' };
	float beta = 0.0f;


	timer.Reset();

	for (int i = 0; i < 1329; i++)
	{
		mkl_scsrmv(&trans, &m, &n, &alpha, matdescra, csr, ja, ia, ia + 1, x, &beta, y);

	}
	std::cout << "Elapsed time of mkl csr-matrix vector multiplication is :" << timer << " ms" << std::endl;
	std::cout << std::accumulate(y, y + m, 0.0f) << "," << *std::max_element(y, y + m)
		<< "," << *std::min_element(y, y + m) << std::endl;

}


int main()
{
	try
	{
		Bench();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
	}

}

