#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


struct par_patch_nod_info 
{
   int mynrpatch;
   int nrpatch;
   vector<Array<int>> vert_contr;
   vector<Array<int>> edge_contr;
   vector<Array<int>> face_contr;
   vector<Array<int>> elem_contr;
   Array<int> patch_natural_order_idx;
   Array<int> patch_global_dofs_ids;
   // constructor
   par_patch_nod_info(ParMesh * cpmesh_, int ref_levels_);
   // Print
   void Print(int rank_id);
private:
   int ref_levels=0;
   ParMesh pmesh;
   FiniteElementCollection *aux_fec=nullptr;
   ParFiniteElementSpace *aux_fespace=nullptr;
};
struct par_patch_dof_info 
{
   MPI_Comm comm = MPI_COMM_WORLD;
   int mynrpatch;
   int nrpatch;
   vector<Array<int>> patch_tdofs;
   vector<Array<int>> patch_local_tdofs;
   // constructor
   par_patch_dof_info(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace);
   void Print();
};

struct par_patch_assembly
{
   MPI_Comm comm;
   int mynrpatch;
   int nrpatch;
   Array<int>tdof_offsets;
   vector<Array<int>> patch_other_tdofs;
   HypreParMatrix * A = nullptr;
   ParFiniteElementSpace *fespace=nullptr;
   // constructor
   par_patch_assembly(ParMesh * cpmesh_, int ref_levels_,ParFiniteElementSpace *fespace_, HypreParMatrix * A_);
   int get_rank(int tdof);
   void compute_trueoffsets();

};


bool its_a_patch(int iv, Array<int> patch_ids);

SparseMatrix * GetDiagColumnValues(const Array<int> & tdof_i, SparseMatrix & diag,
const int * row_start);

void GetOffdColumnValues(const Array<int> & tdof_i, const Array<int> & tdof_j, SparseMatrix & offd, const int * cmap, 
                         const int * row_start, SparseMatrix * PatchMat);

void GetArrayIntersection(const Array<int> & A, const Array<int> & B, Array<int>  & C); 

void GetColumnValues(const int tdof_i,const Array<int> & tdof_j, SparseMatrix & diag ,
SparseMatrix & offd, const int *cmap, const int * row_start, Array<int> &cols, Array<double> &vals);


int GetNumColumns(const int tdof_i, const Array<int> & tdof_j, SparseMatrix & diag,
SparseMatrix & offd, const int * cmap, const int * row_start);

void GetColumnValues2(const int tdof_i,const Array<int> & tdof_j, SparseMatrix & diag ,
SparseMatrix & offd, const int *cmap, const int * row_start, Array<int> &cols, Array<double> &vals);

