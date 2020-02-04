//                                MFEM Example 3
//        IMPLICIT Residual Based Error Estimator for Maxwell Equations 
//
// Compile with: make ex3
//
// Sample runs:  ex3 -m ../data/star.mesh
//               ex3 -m ../data/beam-tri.mesh -o 2
//               ex3 -m ../data/beam-tet.mesh
//               ex3 -m ../data/beam-hex.mesh
//               ex3 -m ../data/escher.mesh
//               ex3 -m ../data/escher.mesh -o 2
//               ex3 -m ../data/fichera.mesh
//               ex3 -m ../data/fichera-q2.vtk
//               ex3 -m ../data/fichera-q3.mesh
//               ex3 -m ../data/square-disc-nurbs.mesh
//               ex3 -m ../data/beam-hex-nurbs.mesh
//               ex3 -m ../data/amr-hex.mesh
//               ex3 -m ../data/fichera-amr.mesh
//               ex3 -m ../data/star-surf.mesh -o 1
//               ex3 -m ../data/mobius-strip.mesh -f 0.1
//               ex3 -m ../data/klein-bottle.mesh -f 0.1
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifdef MFEM_USE_SIMMETRIX
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>


using namespace std;
using namespace mfem;

//---------------------- My Function Declarations----------------------------//
//---------------------------------------------------------------------------//
void equilibrateFluxes(Mesh* mesh, GridFunction x);

// Helper Methods
int checkConnectivity(Array<int> a, Array<int> b); // helper method
void computeFaceOutwardNormal(int faceNo, int elemNo, Vector& normal, Mesh* mesh); // helper method
void normalizeVector(Vector& v); 
void getElemCavityFaces(int elem, Vector ord_fi, Mesh* mesh, Vector& faces);
int edgeIsExterior(int edgeNo, int nf, const int* fi, Mesh* mesh);
void cross3(Vector v1, Vector v2, Vector& cross);
double computeFluxFaceIntegral(int faceNo, int elemNo, int edgeIndex, Mesh* mesh,
                               FiniteElementSpace* fespace, GridFunction x);  
int findEdgeLocalIndex(int edgeNo, int elemNo, Mesh* mesh);
double computeBilinearFormIntegral(int edgeNo, int elemNo, Mesh* mesh, GridFunction x,
                                   FiniteElementSpace* fespace, CurlCurlIntegrator* cci,
                                   VectorFEMassIntegrator* mi);
double computeLinearFormIntegral(int edgeNo, int elemNo, Mesh* mesh, FiniteElementSpace* fespace, VectorFEDomainLFIntegrator* dlfi);
int findPosition(int item, const int* contents, int size);
void computeFaceMassMatrix(int faceNo, Mesh* mesh,
                           FiniteElementSpace* fespace, DenseMatrix& facemat);
void orderEdgePatchEntities(int edgeNo, int ne, int nf, const int* ei, const int* fi,
                            Mesh* mesh, Vector& ord_ei, Vector& ord_fi);
void assembleInteriorLHS(int edgeNo, Vector ord_ei, Vector ord_fi,
                         Mesh* mesh, DenseMatrix& M);
void assembleInteriorRHS(int edgeNo, Vector ord_ei, Vector ord_fi, Mesh* mesh,
                 FiniteElementSpace* fespace, GridFunction x, VectorFEDomainLFIntegrator* dlfi,
                 CurlCurlIntegrator* cci, VectorFEMassIntegrator* mi, Vector& rhs);
double computeFluxTermIntegral(int edgeNo, int elemNo, Vector ord_fi, Mesh* mesh,
                               GridFunction x, FiniteElementSpace* fespace);
void assembleExteriorLHS(int edgeNo, Vector ord_ei, Vector ord_fi,
                         Mesh* mesh, DenseMatrix& M);
void assembleExteriorRHS(int edgeNo, Vector ord_ei, Vector ord_fi, Mesh* mesh,
                   FiniteElementSpace* fespace, GridFunction x, VectorFEDomainLFIntegrator* dlfi,
                   CurlCurlIntegrator* cci, VectorFEMassIntegrator* mi, Vector& rhs);
void evaluateThetaVector(int faceNo, Vector theta_values, Mesh* mesh,
                         FiniteElementSpace* fespace, Vector& theta_vec);
int findEdgeLocalIndexInFace(int edgeNo, int faceNo, Mesh* mesh);
void computeLambda_BVP(int elemNo, FiniteElementSpace* fespace, Mesh* mesh,
                       std::vector<vector<double>> THETA_COEFF, GridFunction x, Vector& lambda_k);
void computeBilinearForm_BVP(int elemNo, Mesh* mesh, GridFunction x, 
                             FiniteElementSpace* fespace, Vector& b_k);
void applyDBC(int elemNo, Mesh* mesh, FiniteElementSpace* fespace, Table edge_face);
void getEssentialElementTrueDofs(int elemNo, const Array<int> &bdr_attr_is_ess,
                                 Array<int> &ess_elem_tdofs_list,
                                 Array<int> &uness_elem_tdofs_list,
                                 Mesh* mesh, FiniteElementSpace* fespace);
void eliminateBCs(DenseMatrix& A, Vector& X, Vector& B, Array<int> ess_tdofs_list, Array<int> uness_tdofs_list,
                  DenseMatrix& Anew, Vector& Bnew);
double computeElementExactL2Error(int elemNo, VectorCoefficient &exsol, Mesh* mesh,
                                  FiniteElementSpace* fespace, GridFunction x);
double computeElementImplicitL2Error(int elemNo, Vector error_coeff, Mesh* mesh,
                                  FiniteElementSpace* fespace);

 

// Testing Methods
void printDenseMatrix(DenseMatrix mat); // (testing method)
//--------------------------------------------------------------------------//
//--------------------------------------------------------------------------//

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI (required by PUMI).
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   
   // 2. Parse command-line options.
   //const char *mesh_file = "../../data/pumi/serial/beam-tet.smb";
   const char *mesh_file = "/users/maliks2/Meshes/MaxwellMeshes/Beam/beam-tet.smb";
   //const char *mesh_file = "/users/maliks2/Meshes/MaxwellMeshes/Fichera/coarse/fichera.smb";
   //const char *mesh_file = "/users/maliks2/Meshes/MaxwellMeshes/Fichera/fine/fichera_fine.smb";
   //const char *mesh_file = "/users/maliks2/Meshes/MaxwellMeshes/Fichera/25k/fichera_25k.smb";
#ifdef MFEM_USE_SIMMETRIX
   //const char *model_file = "../../data/pumi/geom/beam.x_t";
   const char *model_file = "/users/maliks2/Meshes/MaxwellMeshes/Beam/beam.x_t";
   //const char *model_file = "/users/maliks2/Meshes/MaxwellMeshes/Fichera/fichera.x_t";
   //const char *model_file = "/users/maliks2/Meshes/MaxwellMeshes/Fichera/fine/fichera_nat.x_t"; 
   //const char *model_file = "/users/maliks2/Meshes/MaxwellMeshes/Fichera/25k/fichera_25k_nat.x_t"; 
#else
   const char *model_file = "../../data/pumi/geom/Kova.dmg";
#endif
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   int geom_order = 1;
   int ref_levels = 1;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&model_file, "-p", "--parasolid",
                  "Parasolid model to use.");
   args.AddOption(&geom_order, "-go", "--geometry_order",
                  "Geometric order of the model");
   args.AddOption(&ref_levels, "-rf", "--ref_levels",
                  "Uniform refinement levels");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
        args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
     args.PrintOptions(cout);
   }
   kappa = freq * M_PI; // EM

   // 3. Read the SCOREC Mesh
   PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
   Sim_readLicenseFile(0);
   gmi_sim_start();
   gmi_register_sim();
#endif
   gmi_register_mesh();

   apf::Mesh2* pumi_mesh;
   pumi_mesh = apf::loadMdsMesh(model_file, mesh_file);

   // 4. Increase the geometry order if necessary
   if (ref_levels > 1)
   {
      ma::Input* uniInput = ma::configureUniformRefine(pumi_mesh, ref_levels);

      if ( geom_order > 1)
      {
         crv::adapt(uniInput);
      }
      else
      {
         ma::adapt(uniInput);
      }
   }
   /*if (geom_order > 1)
   {
      crv::BezierCurver bc(pumi_mesh, geom_order, 2);
      bc.run();
   }*/
   pumi_mesh->verify();


   // 5. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new PumiMesh(pumi_mesh, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // Print Mesh Stats:
   printf(" INITIAL MESH \n");
   printf("Number of Elements: %d \n", mesh->GetNE());
   printf("Number of Faces: %d \n", mesh->GetNFaces());
   printf("Number of Edges: %d \n", mesh->GetNEdges());
   printf("Number of Vertices: %d \n", mesh->GetNV());

   // 6. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   /*
   {
      int ref_levels = 0;
                    //(int)floor(log(20000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   */
   mesh->ReorientTetMesh(); // EM 

   // Print Mesh Stats
   printf("FINAL MESH \n");
   printf("Number of Elements: %d \n", mesh->GetNE());
   printf("Number of Faces: %d \n", mesh->GetNFaces());
   printf("Number of Edges: %d \n", mesh->GetNEdges());
   printf("Number of Vertices: %d \n", mesh->GetNV());

   // 7. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim, 1, 0);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 8. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   // (SAMI) Basically, all edge dofs on the boundary have been marked as
   //        essential dofs and saved in ess_tdof_list
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   printf("ESS_TDOF_LIST SIZE: %d \n", ess_tdof_list.Size());

   // 9. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.
   VectorFunctionCoefficient f(sdim, f_exact);
   LinearForm *b = new LinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 10. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   GridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, E_exact);
   x.ProjectCoefficient(E);

   // 11. Set up the bilinear form corresponding to the EM diffusion operator
   //    curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //    integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

   // 12. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();
/*
   cout << " Initial b - from assembling the linear form " << endl;
   b->Print(cout,1); cout << endl;

   cout << "Initial x - initialized by projecting the exact solution " << endl;
   x.Print(cout, 1); cout << endl;
*/
   


   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
/*
   cout << "Final b - after eliminating BC " << endl;
   b->Print(cout,1); cout << endl;

   cout << "Final x - hypo that there is no change in initial and final " << endl;
   x.Print(cout, 1); cout << endl;

   cout << "Vector B - after eliminating BC " << endl;
   B.Print(cout,1); cout << endl;

   cout << "Vector X - hypo that there is no change in initial and final " << endl;
   X.Print(cout, 1); cout << endl;
*/
   cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
   // 10. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //     solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 800, 1e-12, 0.0);
#else
   // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   // 11. Recover the solution as a finite element grid function.
      a->RecoverFEMSolution(X, *b, x);
      // x.Save(cout);

   // 12. Compute and print the L^2 norm of the error.
   cout << "\n|| E_h - E ||_{L^2} = " << x.ComputeL2Error(E) << '\n' << endl;

   // 13. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

//--------------------------------------------------------
//--------------------------------------------------------
// Testing Space


int faceNo = 0;
int elem0 = 0;
int elem6 = 6;
if(mesh->FaceIsInterior(faceNo)) 
{
  cout << "Face is Interior " << endl;
  FaceElementTransformations* ftrans = mesh->GetFaceElementTransformations(faceNo);

  /* 1. compute outward unit normal on the face
  Vector normal(3); normal = 0.0;
  int elemNo1 = ftrans->Elem1No;
  int elemNo2 = ftrans->Elem2No; 
  computeFaceOutwardNormal(faceNo, elem0, normal, mesh);
  normalizeVector(normal); 
  cout << "Normal "; normal.Print(cout, 3); cout << endl;
  */
  /* 2. set integration rule on the face and transform it to elem1 and elem2 
  const IntegrationRule *ir = &IntRules.Get(ftrans->FaceGeom, ftrans->Elem1->Order());
  const IntegrationPoint& ip = ir->IntPoint(0); // TODO: include for higher order
  ftrans->Face->SetIntPoint(&ip);
  IntegrationPoint eip1;
  ftrans->Loc1.Transform(ip, eip1);
  ftrans->Elem1->SetIntPoint(&eip1);
  IntegrationPoint eip2;
  ftrans->Loc2.Transform(ip, eip2);
  ftrans->Elem2->SetIntPoint(&eip2);
  */
  
  // 3. Evaluate Shape Functions on element 0
  /*{
    const FiniteElement* fe = fespace->GetFE(elem0);
    DenseMatrix shape_mat(fe->GetDof(), fe->GetDim()); // dof x dim
    fe->CalcPhysCurlShape(*ftrans->Elem1, shape_mat);
    cout << "Element 0 Shape Matrix " << endl; 
    printDenseMatrix(shape_mat);
    cout << endl;

    cout << "Element 0 shared shape functions" << endl;
    Vector s(3); 
    shape_mat.GetRow(3, s); s.Print(cout, 3);
    shape_mat.GetRow(4, s); s.Print(cout, 3);
    shape_mat.GetRow(5, s); s.Print(cout, 3); cout << endl;
   
    Array<int> eldofs;
    fespace->GetElementDofs(elem0, eldofs);
    Vector eldofs_vals(eldofs.Size()); 
    x.GetSubVector(eldofs, eldofs_vals); // TODO fix the negative dof value??
    cout << "Element Dofs for Elem0 "; eldofs.Print(cout, 6);
    cout << "Dof Values for Elem0 "; eldofs_vals.Print(cout, 6);
    
  }*/
  /* 3. Evaluate Shape Functions on element 6
  {
    const FiniteElement* fe = fespace->GetFE(elem6);
    DenseMatrix shape_mat(fe->GetDof(), fe->GetDim()); // dof x dim
    fe->CalcPhysCurlShape(*ftrans->Elem2, shape_mat); 
    cout << "Element 6 Shape Matrix " << endl; 
    printDenseMatrix(shape_mat);
    cout << endl;

    cout << "Element 6 shared shape functions" << endl;
    Vector s(3); 
    shape_mat.GetRow(0, s); s.Print(cout, 3); 
    shape_mat.GetRow(2, s); s.Print(cout, 3);    
    shape_mat.GetRow(4, s); s.Print(cout, 3); cout << endl;         

    Array<int> eldofs;
    fespace->GetElementDofs(elem6, eldofs);
    Vector eldofs_vals(eldofs.Size()); 
    x.GetSubVector(eldofs, eldofs_vals); // TODO fix the negative dof value??
    cout << "Element Dofs for Elem6 "; eldofs.Print(cout, 6);
    cout << "Dof Values for Elem6 "; eldofs_vals.Print(cout, 6);
  }*/
}
 
//const Table &elem_elem = mesh->ElementToElementTable();
/*for (int elemNo = 0; elemNo < mesh->GetNE(); elemNo++)
{
  int nel = elem_elem.RowSize(elemNo);
  const int *ei = elem_elem.GetRow(elemNo); 
  Array<int> dofs;
  fespace->GetElementDofs(elemNo,dofs);
  cout << "Element No " << elemNo << endl;
  cout << "Element Dofs "; dofs.Print(cout, dofs.Size());
  cout << "Adjacent Elements "; for(int i = 0; i < nel; i++) {cout << ei[i] << " ";} cout << endl;
  Array<int> elem_faces, faces_cor;
  mesh->GetElementFaces(elemNo, elem_faces, faces_cor);
  cout << "Downward Faces "; elem_faces.Print(cout, elem_faces.Size());
  Array<int> elemv;
  mesh->GetElementVertices(elemNo, elemv);
  cout << "Element Downward Vertices "; elemv.Print(cout, elemv.Size());
  Array<int> elem_edges, edges_cor;
  mesh->GetElementEdges(elemNo, elem_edges, edges_cor);
  cout << "Downward Edges "; elem_edges.Print(cout, elem_edges.Size());
  cout << "Edge Downward Vertices " << endl;
  for(int i = 0; i < elem_edges.Size(); i++)
  {
    Array<int> edgev;
    mesh->GetEdgeVertices(elem_edges[i], edgev);
    cout << "Downward Edges "; edgev.Print(cout, edgev.Size());
  }

  cout << endl;

}*/


cout << "END TESTING " << endl;

  /////////////////////////////////////////////////////////////////////////
  //---------------------------------------------------------------------//
  //------------ Implicit Residual Error Routine-------------------------//

  //-----FACE AND ELEMENT CONNECTIVITIES OF EDGES-----//   
  // get edge to face connections
  Table &face_edge = *mesh->GetFaceEdgeTable(); 
  Table edge_face; 
  Transpose(face_edge,edge_face); 

  // get edge to element connections
  const Table &elem_edge = mesh->ElementToEdgeTable(); 
  Table edge_elem; 
  Transpose(elem_edge, edge_elem); 

  // get element to element connections
  //const Table &elem_elem = mesh->ElementToElementTable();
  //---------------------------------------------------//

  //-----EQUILIBRATION OF RESIDUALS METHOD-----// 

  // Define Bilinear and Linear Forms 
  VectorFEDomainLFIntegrator* dlfi = new VectorFEDomainLFIntegrator(f);
  CurlCurlIntegrator* cci = new CurlCurlIntegrator(*muinv);
  VectorFEMassIntegrator* mi = new VectorFEMassIntegrator(*sigma);

  // Define a vector of vectors to save the values of g for all edges
  // Basically, for each edge system of eqns, we save the solution vector g.
  std::vector<vector<double>> allgs(mesh->GetNEdges());

  // Define a vector of vectors to save the three values of g 
  // on each face
  std::vector<vector<double>> facegs(mesh->GetNFaces());

  // Define a vector of vectors to save the edges corresponding
  // to g on each face (in order)
  std::vector<vector<double>> edgesOfgs(mesh->GetNFaces());

  int testcount = 0; 
  // loop over all edges to build edge-based patches
  for (int edgeNo = 0; edgeNo < mesh->GetNEdges(); edgeNo++)
  {
    // get connectivity info for each edge (to build edge patch)
    int nf = edge_face.RowSize(edgeNo);
    int ne = edge_elem.RowSize(edgeNo);
    const int *fi = edge_face.GetRow(edgeNo); // faces adjacent to edge i
    const int *ei = edge_elem.GetRow(edgeNo); // elements adjacent to edge i

    // We treat interior edge patches and boundary edge patches in separate loops. 
    // 1: Boundary Edges
    if(1 == edgeIsExterior(edgeNo, nf, fi, mesh))
    {
      cout << endl; cout << " ----------------------------------------------- " << endl;
      cout << "For boundary edge cavity (edgeNo " << edgeNo << ")" << " ne " << ne << " nf " << nf << endl;

      // FIRST, Arrange the upward adjacent faces and elements in an order
      Vector ord_ei, ord_fi; 
      orderEdgePatchEntities(edgeNo, ne, nf, ei, fi, mesh, ord_ei, ord_fi);
      cout << "ordered upward adjacent elements: ";  ord_ei.Print(cout, ne);
      cout << "ordered upward adjacent faces: ";  ord_fi.Print(cout, nf);

      // SECOND, assemble LHS square matrix ((g_k)^2 -> min) (DEMKOWICZ)
      DenseMatrix A(ne+nf,ne+nf);
      assembleExteriorLHS(edgeNo, ord_ei, ord_fi, mesh, A);
      //cout << " Matrix A " << endl; printDenseMatrix(A);

      // THIRD, assemble RHS vector (G)
      Vector G(ne+nf); G = 0.0;
      assembleExteriorRHS(edgeNo, ord_ei, ord_fi, mesh, fespace, x, dlfi, cci, mi, G);
      cout << "G: "; G.Print(cout, ne+nf);
      cout << "Sum G: " << G.Sum() << endl;

      // FOURTH, solve the system A g = G for g
      Vector g(ne+nf); g = 0.0;
      A.Invert();
      A.Mult(G,g);
      cout << "g: "; g.Print(cout, ne+nf); cout <<endl;

      // FIFTH, save the g values in the vector of vectors
      // Ignore the values g(nf)-g(nf+ne) which are associated
      // with Lagrange multipliers.
      allgs[edgeNo].resize(nf);
      for(int ii = 0; ii < nf; ii++)
      {
        allgs[edgeNo][ii] = g(ii);
      }

      // SIXTH, save the g values on their respective faces
      for(int ii = 0; ii < nf; ii++)
      {
        int faceNo = ord_fi(ii);
        facegs[faceNo].push_back(g(ii));
        edgesOfgs[faceNo].push_back(edgeNo);
      } 

    }
    else // 2. Interior edges 
    {
      cout << endl; cout << " ----------------------------------------------- " << endl;
      cout << "For interior edge cavity (edgeNo " << edgeNo << ")" << " ne " << ne << " nf " << nf << endl;

      // FIRST, Arrange the upward adjacent faces and elements in an order
      Vector ord_ei, ord_fi;
      orderEdgePatchEntities(edgeNo, ne, nf, ei, fi, mesh, ord_ei, ord_fi);
      cout << "ordered upward adjacent elements: ";  ord_ei.Print(cout, ne);
      cout << "ordered upward adjacent faces: ";  ord_fi.Print(cout, nf);

      // SECOND, assemble LHS square singular matrix A.
      // Then, transpose it (At). Will need At later.
      // Also, get connectivity matrix (T = A*At) (also singular).
      // TA + 1.0 to pick a particular solution.
      DenseMatrix A(ne,nf);
      assembleInteriorLHS(edgeNo, ord_ei, ord_fi, mesh, A);
      DenseMatrix At(nf,ne);  At.Transpose(A); 
      DenseMatrix T(ne,nf); 
      Mult(A, At, T); 
      DenseMatrix D(ne,nf); D = 1.0;  
      T.Add(1.0,D); 
      //cout << "T matrix " << endl; printDenseMatrix(T);

      // THIRD, assemble the RHS vector (G)
      Vector G(ne); G = 0.0; 
      assembleInteriorRHS(edgeNo, ord_ei, ord_fi, mesh, fespace, x, dlfi, cci, mi, G);
      cout << "initial G: ";  G.Print(cout, ne);
      cout << "Sum G: " << G.Sum() << endl; if(G.Sum() > 0.00001) {testcount++;}

      // FOURTH, make sure sum of all entries of G is zero 
      Vector G_avg(G.Size()); G_avg = G.Sum() / G.Size();
      subtract(G, G_avg, G);
      cout << " final G: ";  G.Print(cout, ne);

      // FIFTH, solve the system T mu = G for mu (mu comes from the Lagrangian).
      Vector mu(ne); mu = 0.0;
      CG(T, G, mu, 0, 10, 1e-12, 1e-24);

      // SIXTH, solve the system g = At mu for g.        
      Vector g(nf); g = 0.0;
      A.MultTranspose(mu,g);
      //cout << "Transpose this matrix " << endl; printDenseMatrix(A); 
      cout << "g: ";  g.Print(cout, ne); cout << endl;

      // SEVENTH, save the g values in the vector of vectors
      allgs[edgeNo].resize(g.Size());
      for(int ii = 0; ii < nf; ii++)
      {
        allgs[edgeNo][ii] = g(ii);
      }

      // SEVENTH SEVENTH, save the g values on their respective faces
      for(int ii = 0; ii < nf; ii++)
      {
        int faceNo = ord_fi(ii);
        facegs[faceNo].push_back(g(ii));
        edgesOfgs[faceNo].push_back(edgeNo);
      } 
    } // end else (interior edges loop)
  } // end loop ( all edge based patches ) 
  // At this point, each face of the mesh has 3 g values. 
  cout << "Problematic Edges " << testcount++ << endl; // test

  //------------------------------------------------//
  //------------------------------------------------//
  // NOW SOLVE FOR THETA ON EACH FACE OF EACH ELEMENT
  // loop over each element

  // Define a vector of vectors to save the theta vectors on all faces
  std::vector<vector<double>> THETA_COEFF(mesh->GetNFaces());
  cout << " Solving for THETA VECTORS " << endl;

  // loop over all faces
  for(int faceNo = 0; faceNo < mesh->GetNFaces(); faceNo++)
  {
    // Get edges of each face
    Array<int> face_edges, edges_cor; 
    mesh->GetFaceEdges(faceNo, face_edges, edges_cor);
    cout << "FaceNo " << faceNo << " Edges: "; face_edges.Print(cout, face_edges.Size());
    cout << "FaceNo " << faceNo << " Edge Or: "; edges_cor.Print(cout, face_edges.Size());
    int num_edges = face_edges.Size();

    // Assemble RHS Vector -- (map to MFEM edge order)
    Vector g(num_edges);
    for(int ii = 0; ii < num_edges; ii++)
    {
      int edgeNo = face_edges[ii];
      int pos = 0;
      for(int j = 0; j < num_edges; j++)
      {
        if(edgeNo == edgesOfgs[faceNo][j]) 
          break;
        else pos++;
      }
      g(ii) = facegs[faceNo][pos];
    }
    cout << "FaceNo " << faceNo << " gs: "; g.Print(cout, g.Size()); cout << endl;
     
    
    // Assemble LHS mass matrix
    DenseMatrix mass_matrix(num_edges); mass_matrix = 0.0; 
    computeFaceMassMatrix(faceNo, mesh, fespace, mass_matrix);
    //cout << endl; cout << "Mass matrix " << endl; printDenseMatrix(mass_matrix); cout << endl;
 
    // Solve the system
    Vector theta(num_edges); theta = 0.0;
    CG(mass_matrix, g, theta, 0, 20, 1e-12, 1e-24);
    // cout << "Theta values for face " << faceNo << endl; theta.Print(cout,num_edges);
 
    // Save all theta parameters on their corresponding faces
    for(int ii = 0; ii < theta.Size(); ii++)
    {
      THETA_COEFF[faceNo].push_back(theta(ii));
    }
  }
  // END OF EQUILIBRATION OF RESIDUALS METHOD // 
  // ---------------------------------------- // 
  
  // Testing Space - Checking to see if equilibration condition
  // is satisfied.  
  /*cout << " TESTING EQUILIBRATION CONDITION " << endl;  
  for(int elemNo = 0; elemNo < mesh->GetNE(); elemNo++)
  {
    cout << "Element " << elemNo << endl;
    const FiniteElement* fel = fespace->GetFE(elemNo);
    int neldofs = fel->GetDof();
    int dim = fel->GetDim();
    int order = fel->GetOrder(); 
    
    // Get element faces
    Array<int> elem_faces, faces_or;
    mesh->GetElementFaces(elemNo, elem_faces, faces_or);

    Vector result(neldofs); result = 0.0;
    // loop over all element faces to compute face integrals
    for(int ff = 0; ff < elem_faces.Size(); ff++)
    {
      int faceNo = elem_faces[ff];
      FaceElementTransformations* ftr = mesh->GetFaceElementTransformations(faceNo);
      int elemNo1 = ftr->Elem1No;
      int elemNo2 = ftr->Elem2No; 

      const FiniteElement* facefe = fespace->GetFaceElement(faceNo);
      int nfdofs = facefe->GetDof();
      DenseMatrix facevshape(nfdofs, dim); facevshape = 0.0; // dof x dim (3 x 3)

      Array<int> facedofs; fespace->GetFaceDofs(faceNo, facedofs);
      //cout << " ND Face " << faceNo << " Dofs "; facedofs.Print(cout, facedofs.Size()); 

      // get theta coefficients on each face
      Vector theta_coeff(dim);
      for(int i = 0; i < dim; i++) { theta_coeff(i) = THETA_COEFF[faceNo][i]; }

      // Compute integral on the face 
      Vector faceint(neldofs); faceint = 0.0;
      const IntegrationRule *ir = &IntRules.Get(ftr->FaceGeom, 2*order);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
        const IntegrationPoint &ip = ir->IntPoint(i);
        ftr->Face->SetIntPoint(&ip);
        IntegrationPoint eip1;
        ftr->Loc1.Transform(ip, eip1);
        ftr->Elem1->SetIntPoint(&eip1);
        IntegrationPoint eip2;
        ftr->Loc2.Transform(ip, eip2);
        if(mesh->FaceIsInterior(faceNo))
        { 
          ftr->Elem2->SetIntPoint(&eip2);
        }

        // evaluate theta vector using ip
        Vector theta_vec(dim); theta_vec = 0.0;
        facefe->CalcVShape(*ftr->Face, facevshape);
        for(int i = 0; i < theta_coeff.Size(); i++)
        {
          Vector shape;
          facevshape.GetRow(i, shape);
        
          if(facedofs[i] < 0)
            shape.Neg();

          theta_vec.Add(theta_coeff(i), shape); 
        }
        if(elemNo == elemNo2) { theta_vec.Neg(); }
        cout << "Theta Vector for FaceNo " << faceNo << ": "; theta_vec.Print(cout, theta_vec.Size()); 
        //cout << "FaceVShape " << faceNo << endl; printDenseMatrix(facevshape); 
       
        
        if(elemNo == elemNo1)
        {
          const FiniteElement* fel1 = fespace->GetFE(elemNo1);
          DenseMatrix elemvshape(fel1->GetDof(), fel1->GetDim()); // dof x dim
          fel1->CalcVShape(*ftr->Elem1, elemvshape); 

          // negating to take care of negative dofs in Nedelec spaces
          Array<int> eldofs; fespace->GetElementDofs(elemNo, eldofs);
          for(int ind = 0; ind < eldofs.Size(); ind++)
          {
            if(eldofs[ind] < 0)
            {
              for(int j = 0; j < dim; j++) 
                {elemvshape(ind,j) = -1*elemvshape(ind,j);}
            }
          }

          Vector temp(neldofs); temp = 0.0;   
          elemvshape.Mult(theta_vec, temp);
          temp *= ip.weight * ftr->Face->Weight();
          faceint.Add(1.0, temp);
        }
        else
        {
          const FiniteElement* fel2 = fespace->GetFE(elemNo2);
          DenseMatrix elemvshape(fel2->GetDof(), fel2->GetDim()); // dof x dim
          fel2->CalcVShape(*ftr->Elem2, elemvshape); 

          // negating to take care of negative dofs defined by Nedelec spaces
          Array<int> eldofs; fespace->GetElementDofs(elemNo, eldofs);
          for(int ind = 0; ind < eldofs.Size(); ind++)
          {
            if(eldofs[ind] < 0)
            {
              for(int j = 0; j < dim; j++) 
                {elemvshape(ind,j) = -1*elemvshape(ind,j);}
            }
          }

          Vector temp(neldofs); temp = 0.0;   
          elemvshape.Mult(theta_vec, temp);
          temp *= ip.weight * ftr->Face->Weight();
          faceint.Add(1.0, temp);
        }
      } // end int point loop
      cout << "FaceInt for FaceNo " << faceNo << ": "; faceint.Print(cout, neldofs);
      result.Add(1.0, faceint);
    } // end face loop
    cout << "Result for FaceNo " << faceNo << ": "; result.Print(cout, neldofs);
  } // end elem loop */
  cout << " END EQUILIBRATION TESTING " << endl;
  


  // -------------- BVP ------------ //
  // Solving Boundary Value Problems //
  
  // Define PUMI Fields for Visualization In Paraview
  apf::Field* error_Vector_Field = 0;
  apf::Field* error_Scalar_Field = 0;
  apf::Field* E_exact_Field = 0;
  apf::Field* E_fem_Field = 0;
  apf::Field* E_fem_exact_Field = 0;
  apf::Field* error_Index_Field = 0;
  apf::Field* exact_error_Field = 0;
  apf::Field* l2_error_Field = 0;
  apf::Field* exact_error_nodal_Field = 0;
  apf::Field* l2_error_nodal_Field = 0;
  apf::Field* E_exact_nodal_Field = 0;
  apf::Field* E_fem_nodal_Field = 0;

  error_Vector_Field = apf::createIPField(pumi_mesh, "error_Vector_field", apf::VECTOR, 1);
  error_Scalar_Field = apf::createIPField(pumi_mesh, "error_Scalar_field", apf::SCALAR, 1);
  E_exact_Field = apf::createIPField(pumi_mesh, "E_exact_field", apf::VECTOR, 1);
  E_fem_Field = apf::createIPField(pumi_mesh, "E_fem_field", apf::VECTOR, 1);
  E_fem_exact_Field = apf::createIPField(pumi_mesh, "E_fem_exact_field", apf::SCALAR, 1);
  error_Index_Field = apf::createIPField(pumi_mesh, "Effectivity_Index_Field", apf::SCALAR, 1);
  exact_error_Field = apf::createIPField(pumi_mesh, "exact_error_field", apf::SCALAR, 1);
  l2_error_Field = apf::createIPField(pumi_mesh, "l2_error_field", apf::SCALAR, 1);
  exact_error_nodal_Field = apf::createField(pumi_mesh, "exact_error_nodal_field", apf::SCALAR, apf::getLagrange(1));
  l2_error_nodal_Field = apf::createField(pumi_mesh, "l2_error_nodal_field", apf::SCALAR, apf::getLagrange(1));
  E_exact_nodal_Field = apf::createField(pumi_mesh, "E_exact_nodal_field", apf::VECTOR, apf::getLagrange(1));
  E_fem_nodal_Field = apf::createField(pumi_mesh, "E_fen_nodal_field", apf::VECTOR, apf::getLagrange(1));
 
  
  // Define higher-order Nedelec FE Space
  int orderp1 = order+1; cout << "Orderp1 " << orderp1 << endl;
  FiniteElementCollection *fecp1 = new ND_FECollection(orderp1, dim, 1, 0);
  FiniteElementSpace *fespacep1 = new FiniteElementSpace(mesh, fecp1);

  Vector lsum(20); lsum = 0.0;
  Vector bklksum(20); bklksum = 0.0;
  cout << "BVP Loop " << endl;
  // BVP - Element Loop
  apf::MeshEntity* ent;
  apf::MeshIterator* itr = pumi_mesh->begin(3);
  int elemNo = 0;
  while ((ent = pumi_mesh->iterate(itr)))  // iterate over all regions
  {   
    cout << "Element No " << elemNo << endl; 
    // 1. Assemble LHS element matrix
    const FiniteElement* fe = fespacep1->GetFE(elemNo);
    ElementTransformation* eltr = mesh->GetElementTransformation(elemNo);
    int ndofs = fe->GetDof();
    Array<int> eldofs; // Negative ND Dofs 
    fespacep1->GetElementDofs(elemNo, eldofs);

    DenseMatrix elmat_curl(ndofs, ndofs); elmat_curl = 0.0;
    cci->AssembleElementMatrix(*fe, *eltr, elmat_curl);
    DenseMatrix elmat_mass(ndofs, ndofs); elmat_mass = 0.0;
    mi->AssembleElementMatrix(*fe, *eltr, elmat_mass);
    DenseMatrix elmat(ndofs, ndofs); elmat = 0.0;
    Add(elmat_curl, elmat_mass, 1.0, elmat);
    cout << "ElemNo " << elemNo << " : "; eldofs.Print(cout, eldofs.Size()); cout << endl;
    cout << "Initial LHS Matrix for ElemNo " << elemNo << endl; printDenseMatrix(elmat); cout << endl;

    // loop over all entries of elmat to take care of negative dof indices
    int s, t;
    for(int i = 0; i < ndofs; i++)
    {
      if(eldofs[i] < 0) {s = -1;}
      else {s = 1;} 
      for(int j = 0; j < ndofs; j++)
      {
        if(eldofs[j] < 0) {t = -s;}
        else {t = s;} 

        if(t < 0) { elmat(i,j) = -elmat(i,j); }      
      } 
    }
    cout << "Modified LHS Matrix for ElemNo " << elemNo << endl; printDenseMatrix(elmat); cout << endl;

    // 2. Assemble RHS Vector 
    // (i). b_k(E_h, F)  
    Vector b_k;
    computeBilinearForm_BVP(elemNo, mesh, x, fespacep1, b_k);
    cout << "Bilinear Form (2) Vector " << elemNo << ": "; b_k.Print(cout,ndofs); 

    // (ii). l_k(F)
    Vector l_k(ndofs); l_k = 0.0;
    dlfi->AssembleRHSElementVect(*fe, *eltr, l_k);
    for(int i = 0; i < eldofs.Size(); i++)
    {
      if(eldofs[i] < 0)
       l_k(i) =  -1 * l_k(i); // TODO taking care of negative dof value
    }
    //cout << "Linear Form Vector " << elemNo << endl; l_k.Print(cout,1); cout << endl;
    bklksum.Add(1.0, b_k);
    bklksum.Add(-1.0, l_k);

    // (iii). lambda_K(F)
    Vector lambda_k;
    computeLambda_BVP(elemNo, fespacep1, mesh, THETA_COEFF, x, lambda_k); 
    lsum.Add(1.0, lambda_k);  
    //cout << "Flux functionals Vector " << elemNo << endl; lambda_k.Print(cout,1); cout << endl;
    
    // (iv). b_k(E_h, F) - l_k(F) - lambda_k(F)
    Vector rhs(ndofs); rhs = 0.0;
    rhs.Add(1.0, b_k);
    rhs.Add(-1.0, l_k);
    rhs.Add(-1.0, lambda_k);
    //cout << "RHS " << elemNo << endl; rhs.Print(cout,1); cout << endl;
     
    // 3. Apply Dirichlet Boundary Conditions (error = 0 on Dirichlet boundary)
    // get DB dofs
    Array<int> ess_bdr(mesh->bdr_attributes.Max()); ess_bdr = 1;
    Array<int> ess_elem_tdofs_list, uness_elem_tdofs_list;
    getEssentialElementTrueDofs(elemNo, ess_bdr, ess_elem_tdofs_list, uness_elem_tdofs_list, mesh, fespacep1);
    cout << " TRUE DOF LIST FOR ELEM " << elemNo << ": " << ess_elem_tdofs_list.Size() << endl; ess_elem_tdofs_list.Print(cout, ess_elem_tdofs_list.Size()); cout << endl;
    cout << " OTHER DOF LIST FOR ELEM " << elemNo << ": " << uness_elem_tdofs_list.Size() << endl; uness_elem_tdofs_list.Print(cout, uness_elem_tdofs_list.Size()); cout << endl;
  
    // test 
    cout << "LHS " << endl;
    printDenseMatrix(elmat);
    cout << "RHS " << endl; rhs.Print(cout, rhs.Size()); 
 
    // eliminate DBC
    DenseMatrix Anew; Vector Bnew, Xnew, X;
    X.SetSize(ndofs); X = 0.0; // initialize X with exact Dirichlet value (e = 0.0)
    Xnew.SetSize(uness_elem_tdofs_list.Size()); Xnew = 0.0;
    eliminateBCs(elmat, X, rhs, ess_elem_tdofs_list, uness_elem_tdofs_list, Anew, Bnew); 

    // Solve the reduced system
    CG(Anew, Bnew, Xnew, 0, 500, 1e-12, 1e-24); 
    cout << "Reduced Solution Vector Xnew" << elemNo << endl; Xnew.Print(cout,1); cout << endl;

    // Recover solution
    Vector phi(ndofs); // Recover Solution
    for(int i = 0; i < ess_elem_tdofs_list.Size(); i++)
    {  int index = ess_elem_tdofs_list[i];
       phi(index) = X[index];
    }
    for(int i = 0; i < uness_elem_tdofs_list.Size(); i++)
    {  int index = uness_elem_tdofs_list[i];
       phi(index) = Xnew[i];
    }
    cout << "Solution Vector phi " << elemNo << endl; phi.Print(cout,1); cout << endl;
      
    // 5. Evaluate error function on center of each element K
    Geometry::Type geo_type = eltr->GetGeometryType();
    const IntegrationPoint *center = &Geometries.GetCenter(geo_type); // center of elemNo
    eltr->SetIntPoint(center);
    DenseMatrix vshape(ndofs, dim); 
    fe->CalcVShape(*eltr, vshape);
    for(int i = 0; i < eldofs.Size(); i++)
    {
      if(eldofs[i] < 0)
      {
        for(int j = 0; j < dim; j++)
        {
          vshape(i,j) = -vshape(i,j);
        }
      }
    }
      
    Vector verror(dim); verror = 0.0;
    vshape.MultTranspose(phi, verror);   
    //cout << "Error Vector " << endl; verror.Print(cout, 1); cout << endl;

    // 6. Compute l2 Norm Error
    //cout << "Error Norm for element " << elemNo << " :         "  << verror.Norml2() << endl;
    setScalar(error_Scalar_Field, ent, 0, verror.Norml2()); // set error scalar
    Vector3 err; 
    for(int i = 0; i < dim; i++) { err[i] = verror(i); }
    setVector(error_Vector_Field, ent, 0, err); // set error vector


    // 7. Compute L2 Norm Integral Error
    double elem_l2integ_error = computeElementImplicitL2Error(elemNo, phi, mesh, fespacep1);
    setScalar(l2_error_Field, ent, 0, elem_l2integ_error); // set error scalar
  

    elemNo++;
}
  cout << "LAMBDA K SUM " << endl; lsum.Print(cout, 1); cout << endl;    
  cout << "BKLK  K SUM " << endl; bklksum.Print(cout, 1); cout << endl;    


  // Write out PUMI Fields for Paraview Visualization
  //apf::MeshEntity* ent;
  itr = pumi_mesh->begin(3);
  elemNo = 0;
  while ((ent = pumi_mesh->iterate(itr)))  // iterate over all regions
  {
    // 1. write E_fem field (vector)
    //const FiniteElement* fe = fespacep1->GetFE(elemNo);
    ElementTransformation* eltr = mesh->GetElementTransformation(elemNo);
    Geometry::Type geo_type = eltr->GetGeometryType();
    const IntegrationPoint *center = &Geometries.GetCenter(geo_type); // center of elemNo
    Vector vval(dim);
    x.GetVectorValue(elemNo, *center, vval); // get E_h
    Vector3 efem; 
    for(int i = 0; i < dim; i++) { efem[i] = vval(i); }
    setVector(E_fem_Field, ent, 0, efem); 


    // 2. write E_exact field (vector)
    Vector vexact(dim);
    eltr->SetIntPoint(center);
    E.Eval(vexact, *eltr, *center);
    Vector3 eexact; 
    for(int i = 0; i < dim; i++) { eexact[i] = vexact(i); }
    setVector(E_exact_Field, ent, 0, eexact); 
    

    // 3. write E_fem_exact_field (scalar)
    Vector3 efem_exact = efem - eexact;
    double efem_exact_norm = (efem_exact[0] * efem_exact[0]) + (efem_exact[1] * efem_exact[1]) + (efem_exact[2] * efem_exact[2]);
    efem_exact_norm = sqrt(efem_exact_norm);
    setScalar(E_fem_exact_Field, ent, 0, efem_exact_norm); 

    // 4. Write out exact error field
    double exact_elem_error = computeElementExactL2Error(elemNo, E, mesh, fespace, x);
    setScalar(exact_error_Field, ent, 0, exact_elem_error);   


    elemNo++;    
  }

  // Convert Lagrange Constant Error Fields to Nodal Fields
  itr = pumi_mesh->begin(0); // vertices
  while ((ent = pumi_mesh->iterate(itr)))  // iterate over all vertices 
  {
    apf::Adjacent elements;
    pumi_mesh->getAdjacent(ent, 3, elements);  

    // 1. Write Error Nodal Fields (exact, l2)
    // get scalar errors on adjacent elements
    double exact_err_adj_elements[elements.getSize()];
    double l2_err_adj_elements[elements.getSize()];
    for (std::size_t i=0; i < elements.getSize(); ++i){
      exact_err_adj_elements[i] = apf::getScalar(exact_error_Field, elements[i], 0);
      l2_err_adj_elements[i] = apf::getScalar(l2_error_Field, elements[i], 0);
    }

    // compute average
    double exact_average = 0.0;
    double l2_average = 0.0;
    for(int i = 0; i < elements.getSize(); i++){
      exact_average += exact_err_adj_elements[i]; 
      l2_average += l2_err_adj_elements[i]; 
    }
    exact_average = exact_average/elements.getSize();
    l2_average = l2_average/elements.getSize();
    
    // set nodal values
    apf::setScalar(exact_error_nodal_Field, ent, 0, exact_average);
    apf::setScalar(l2_error_nodal_Field, ent, 0, l2_average);

    // 2. Write Solution Nodal Fields (exact, FEM)
    // E exact nodal field (Vector)
    Vector exact_field_average(3); exact_field_average = 0.0;
    apf::Vector3 exact_field;
    for (std::size_t i=0; i < elements.getSize(); ++i){
      apf::getVector(E_exact_Field, elements[i], 0, exact_field);
      for(int j = 0; j < exact_field_average.Size(); j++) {
        exact_field_average(j) += exact_field[j];
      }
    }
    exact_field_average /= elements.getSize();
    Vector3 exact_nodal_field_vec; for(int i = 0; i < 3; i++) { exact_nodal_field_vec[i] = exact_field_average(i);}
    apf::setVector(E_exact_nodal_Field, ent, 0, exact_nodal_field_vec);
    
    // E fem nodal field (Vector)
    Vector fem_field_average(3); fem_field_average = 0.0;
    apf::Vector3 fem_field;
    for (std::size_t i=0; i < elements.getSize(); ++i){
      apf::getVector(E_fem_Field, elements[i], 0, fem_field);
      for(int j = 0; j < fem_field_average.Size(); j++) {
        fem_field_average(j) += fem_field[j];
      }
    }
    fem_field_average /= elements.getSize();
    Vector3 fem_nodal_field_vec; for(int i = 0; i < 3; i++) { fem_nodal_field_vec[i] = fem_field_average(i);}
    apf::setVector(E_fem_nodal_Field, ent, 0, fem_nodal_field_vec);

  }

  apf::writeVtkFiles("nodal_Fields", pumi_mesh);
  //------------------------------------------ FIELDS WRITTEN ----------------------------------------------------//
  cout << " END ROUTINE " << endl;
 
/*
  cout << "PRINTING ALL G's" << endl; 
  for (int i = 0; i < mesh->GetNFaces(); i++)
  {
    if(mesh->FaceIsInterior(i)) { cout << "Interior Face "; }
    else { cout << "Boundary Face "; }
    for (int j = 0; j < facegs[i].size(); j++)
    {
      cout << facegs[i][j] << " ";
    }
    cout << "end" << endl;
  }
*/
  cout << "PRINTING ALL THETA VECTORS ON FACES" << endl; 
  for (int i = 0; i < mesh->GetNFaces(); i++)
  {
    if(mesh->FaceIsInterior(i)) { cout << "Interior Face "; }
    else { cout << "Boundary Face "; }
    for (int j = 0; j < THETA_COEFF[i].size(); j++)
    {
      cout << THETA_COEFF[i][j] << " ";
    }
    cout << "end" << endl;
  }

/*
    cout << "Corresponding edge for each g value: ";
    for (int j = 0; j < edgesOfgs[i].size(); j++)
    {
      cout << edgesOfgs[i][j] << " ";
    }
    cout << "end" << endl;
  }
  cout << "END PRINTING" << endl;
*/     
  //-------------------------------------------------------------------------//
  //-------------End Implicit Residual Error Routine-------------------------//
  //-------------------------------------------------------------------------//

   // 15. Free the used memory.
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
////////-----------------------------------------------//////
////////--------MY FUNCTION DECLARATIONS---------------//////

/**
 * This method checks if any element of array2 is equal to any element of array1.
 * If it finds one similar element between both arrays, it returns 1. Otherwise 0.
 * OPTIMIZE LATER - (Consider using hashset instead of set).
 *
 */
int checkConnectivity(Array<int> a, Array<int> b)
{
  int size = a.Size();
  std::set<int> s;
  for(int i = 0; i < size; i++)
  {
    s.insert(a[i]); 
  }
  
  for(int j = 0; j < size; j++)
  {
    if(s.count(b[j]) == 1)
    {
      return 1; 
    }
  }
  return 0;
}

/**
 * This method computes outward face normal depending on the element with the face.
 * By default, in MFEM, the normal is oriented from elem1 (element with smaller index)
 * to elem2 (element with bigger index). So, we need the extra information of the current
 * element to orient the normal vector outwards. Elem2No = -1 for boundary faces. 
 *
 * @param faceNo face on which to compute outward normal
 * @param elemNo element relative to which 'outward' is decided
 * @param normal 
 * @param mesh
 *
 * FUTURE TODO: compute outward normal for curved surfaces
 * 
 */ 
void computeFaceOutwardNormal(int faceNo, int elemNo, Vector& normal, Mesh* mesh)
{
   FaceElementTransformations* trans = mesh->GetFaceElementTransformations(faceNo);
   Geometry::Type geo_type = mesh->GetFaceGeometryType(trans->Face->ElementNo);
   const IntegrationPoint *center = &Geometries.GetCenter(geo_type); // center of face
   trans->Face->SetIntPoint(center);
   CalcOrtho(trans->Face->Jacobian(), normal); // compute normal
   if((trans->Elem1No != elemNo) && (trans->Elem2No != -1)) 
   {
     normal.Neg();
   }
}

/**
 * This method normalizes a vector using l2 norm.
 */
void normalizeVector(Vector& v)
{
  double l2norm = v.Norml2();
  for(int i = 0; i < v.Size(); i++) 
  {
    v(i) = v(i) / l2norm;   
  }
}

/**
 * This method assembles the RHS vector on each interior
 * edge-based patch.
 *
 * @param edgeNo the edge around which the patch is formed 
 * @param ord_ei elements in the edge-based patch (ordered)
 * @param ord_fi faces in the edge-based patch (ordered)
 * @param    rhs RHS vector to assemble
 * 
 */
void assembleInteriorRHS(int edgeNo, Vector ord_ei, Vector ord_fi, Mesh* mesh,
                   FiniteElementSpace* fespace, GridFunction x, VectorFEDomainLFIntegrator* dlfi,
                   CurlCurlIntegrator* cci, VectorFEMassIntegrator* mi, Vector& rhs)
{
  rhs = 0.0;
  double test = 0.0; // test
  double test2 = 0.0; // test
  int ne = ord_ei.Size();
  int nf = ord_fi.Size();
  cout << "edgeNo " << edgeNo << endl;
  for(int e = 0; e < ne; e++) // loop over all elements in the patch
  {
    // Vector for three RHS terms: (1) Bilinear Form (2) Linear Form (3) Flux Term Integral
    Vector terms3(3); terms3 = 0.0; 

    int elemNo = ord_ei[e];
    // (Step 1) *** compute bilinear form integral (1) *** //
    terms3(0) = computeBilinearFormIntegral(edgeNo, elemNo, mesh, x, fespace, cci, mi);

    // (Step 2) *** compute linear form integral (2) *** //
    terms3(1) = computeLinearFormIntegral(edgeNo, elemNo, mesh, fespace, dlfi);
    test2 += terms3(0) - terms3(1);
        
    // (Step 3) *** compute flux term integral (3) *** //
    terms3(2) = computeFluxTermIntegral(edgeNo, elemNo, ord_fi, mesh, x, fespace);
    test += terms3(2);

    // Step (4) Sum up all values in terms3 and set the sum as the corresponding entry in RHS vector
    rhs(e) = terms3(0) - terms3(1) - terms3(2);
  } 
  cout << "Flux Sum " << test << endl; // test 
  cout << "Bilinear - Linear Sum " << test2 << endl; // test
}

/**
 * This method assembles the RHS vector on each boundary
 * edge-based patch.
 */
void assembleExteriorRHS(int edgeNo, Vector ord_ei, Vector ord_fi, Mesh* mesh,
                   FiniteElementSpace* fespace, GridFunction x, VectorFEDomainLFIntegrator* dlfi,
                   CurlCurlIntegrator* cci, VectorFEMassIntegrator* mi, Vector& rhs)
{
  rhs = 0.0;
  double test = 0.0; // test
  double test2 = 0.0; // test
  int ne = ord_ei.Size();
  int nf = ord_fi.Size();
  cout << "edgeNo " << edgeNo << endl;
  for(int e = 0; e < ne; e++) // loop over all elements in the patch
  {
    // Vector for three RHS terms: (1) Bilinear Form (2) Linear Form (3) Flux Term Integral
    Vector terms3(3); terms3 = 0.0; 

    int elemNo = ord_ei[e];
    // (Step 1) *** compute bilinear form integral (1) *** //
    terms3(0) = computeBilinearFormIntegral(edgeNo, elemNo, mesh, x, fespace, cci, mi);

    // (Step 2) *** compute linear form integral (2) *** //
    terms3(1) = computeLinearFormIntegral(edgeNo, elemNo, mesh, fespace, dlfi);
    test2 += terms3(0) - terms3(1);
    
    // (Step 3) *** compute flux term integral (3) *** //
    terms3(2) = computeFluxTermIntegral(edgeNo, elemNo, ord_fi, mesh, x, fespace);
    test += terms3(2);

    // Step (4) Sum up all values in terms3 and set the sum as the corresponding entry in RHS vector
    rhs(e+nf) = terms3(0) - terms3(1) - terms3(2);
  } 
  cout << "Flux Sum " << test << endl; // test 
  cout << "Bilinear - Linear Sum " << test2 << endl; // test
}

/**
 * This method takes as input an element in the edge patch.
 * It returns the two faces of the given element in the patch.
 * These two faces are adjacent to the edge around which the 
 * patch is formed. 
 *
 * @param elemNo element of which we want to find the faces in the patch
 * @param ord_fi upward adjacent faces of the edge around which the patch is formed
 * @param   mesh input mesh
 */
void getElemCavityFaces(int elemNo, Vector ord_fi, Mesh* mesh, Vector& faces)
{
  faces.SetSize(2);
  int nf = ord_fi.Size();
  int count = 0;
  Array<int> elem_fcs, elem_cor;
  // get the four downward faces of the element
  mesh->GetElementFaces(elemNo, elem_fcs, elem_cor); 
  // for each downward face of the element check 
  // if that face is also amongst the faces in 
  // the edge cavity. If so save it. 
  for(int i = 0; i < elem_fcs.Size(); i++)
  {
    for(int j = 0; j < nf; j++)
    {
      if(elem_fcs[i] == ord_fi(j))
      {
        faces(count) = elem_fcs[i];
        count++;     
      }
    }    
  }
}
/**
 * This method determines if a given edge in the mesh lies in the 
 * interior of the mesh or on the boundary of the mesh. 
 *
 * @param edgeNo given edge
 * @param     nf number of faces adjacent to the edge
 * @param    *fi faces adjacent to the edge
 * @param   mesh input mesh
 * return 1 if edge in on the boundary, 0 otherwise 
 * 
 */
int edgeIsExterior(int edgeNo, int nf, const int* fi, Mesh* mesh)
{
  int edgeLocation = 0;
  for(int i = 0; i < nf; i++)
  {
    if(!(mesh->FaceIsInterior(fi[i]))) // if adjacent face on the boundary
    {   
      return 1;
    }
  } 
  return 0;
}

/*
 * Find the location of a given item in the array.
 *
 * @param     item want to find the position of this item
 * @param contents look in these contents to find the item
 * @param     size size of container
 * return        s location of the item
 *              -1 if item not found
 */
int findPosition(int item, const int* contents, int size)
{
  for(int s = 0; s < size; s++)
  {
    if(item == contents[s]) {return s;}
  }
  cout << "Item not found" << endl;
  return -1;
}

/**
 * This method computes the cross product of two 3x3 vectors
 *
 * @param    v1 
 * @param    v2
 * return cross
 */
void cross3(Vector v1, Vector v2, Vector& cross)
{
  cross[0] = v1[1] * v2[2] - v1[2] * v2[1];
  cross[1] = v1[2] * v2[0] - v1[0] * v2[2];
  cross[2] = v1[0] * v2[1] - v1[1] * v2[0];
  // cout << cross[0] << " " << cross[1] << " " << cross[2] << endl;
}

/**
 * This method computes the averaged flux vector t_k on a face.
 * Then, it computes the flux term integral
 * @param faceNo face on which to compute the flux
 * @param   mesh input mesh
 * @param      x gridfunction containing the solution dofs
 * return    t_k averaged flux vector
 * TODO      include t_k vector calculation on Neumann boundary face
 */
double computeFluxFaceIntegral(int faceNo, int elemNo, int edgeIndex, Mesh* mesh, FiniteElementSpace* fespace, GridFunction x)  
{
  FaceElementTransformations* ftrans = mesh->GetFaceElementTransformations(faceNo);
  // 1. compute outward unit normal on the face
  Vector normal(3); normal = 0.0;
  int elemNo1 = ftrans->Elem1No;
  int elemNo2 = ftrans->Elem2No; 
  computeFaceOutwardNormal(faceNo, elemNo, normal, mesh);
  normalizeVector(normal); 

  // 2. set integration rule on the face and transform it to elem1 and elem2 
  const IntegrationRule *ir = &IntRules.Get(ftrans->FaceGeom, ftrans->Elem1->Order()-1);
  const IntegrationPoint& ip = ir->IntPoint(0); // TODO: include for higher order // see how theta vectors are calculated
  ftrans->Face->SetIntPoint(&ip);
  IntegrationPoint eip1;
  ftrans->Loc1.Transform(ip, eip1);
  ftrans->Elem1->SetIntPoint(&eip1);
  IntegrationPoint eip2;
  ftrans->Loc2.Transform(ip, eip2);
  if(mesh->FaceIsInterior(faceNo))
  { 
    ftrans->Elem2->SetIntPoint(&eip2);
  }

  // 3. compute curl of the solution and t_k vector (interelement averaged flux)
  Vector t_k(3); t_k = 0.0;
  if(mesh->FaceIsInterior(faceNo))
  {
    normal /= 2.0;
    Vector curl1, curl2; 
    x.GetCurl(*ftrans->Elem1, curl1);         
    x.GetCurl(*ftrans->Elem2, curl2);         
    Vector sumCurl(3); sumCurl = 0.0;
    sumCurl.Add(1.0, curl1);
    sumCurl.Add(1.0, curl2);
    cross3(normal, sumCurl, t_k); // get t_k
    cout << "(Interior Face) t_k for FaceNo " << faceNo << " and Elem No " << elemNo << endl;
    //t_k.Print(cout, t_k.Size());
    cout << "normal "; normal.Print(cout, t_k.Size());
  }
  else // Dirichlet face on the boundary
  {
    Vector curl1; 
    x.GetCurl(*ftrans->Elem1, curl1);         
    cross3(normal, curl1, t_k); // get t_k
    cout << "(Boundary Face) t_k for FaceNo " << faceNo << " and Elem No " << elemNo << endl;
    //t_k.Print(cout, t_k.Size());
  }

  // 4. Compute flux face integral
  if(elemNo == elemNo1)
  {
    const FiniteElement* fe = fespace->GetFE(elemNo1);
    DenseMatrix shape_mat(fe->GetDof(), fe->GetDim()); // dof x dim
    fe->CalcVShape(*ftrans->Elem1, shape_mat); 
    Vector edgeShape(3); edgeShape = 0.0;
    shape_mat.GetRow(edgeIndex, edgeShape);

    // negating to take care of negative dofs defined by Nedelec spaces
    Array<int> eldofs; fespace->GetElementDofs(elemNo, eldofs);
    if(eldofs[edgeIndex] < 0) { edgeShape.Neg(); }

    cout << "(1) tk "; t_k.Print(cout, 3); // test
    cout << "(1) sh with edgeIndex " << edgeIndex << ": "; edgeShape.Print(cout, 3);  // test
    double integral_value = (t_k * edgeShape) * ip.weight * ftrans->Face->Weight();
    return integral_value;
  }
  else // if elemNo == elemNo2
  {
    const FiniteElement* fe = fespace->GetFE(elemNo2);
    DenseMatrix shape_mat(fe->GetDof(), fe->GetDim()); // dof x dim
    fe->CalcVShape(*ftrans->Elem2, shape_mat); 
    Vector edgeShape(3); edgeShape = 0.0;
    shape_mat.GetRow(edgeIndex, edgeShape);

    // negating to take care of negative dofs defined by Nedelec spaces // TODO can use edge orientation for this directly
    Array<int> eldofs; fespace->GetElementDofs(elemNo, eldofs);
    if(eldofs[edgeIndex] < 0) { edgeShape.Neg(); } 

    cout << "(2) tk "; t_k.Print(cout, 3); // test
    cout << "(2) sh with edgeIndex " << edgeIndex << ": "; edgeShape.Print(cout, 3); // test
    double integral_value = (t_k * edgeShape) * ip.weight * ftrans->Face->Weight();
    return integral_value;
  }
}

/**
 * This function gets the local index of a given edge in
 * an element adjacent to the edge. We need this to help 
 * pick the appropriate shape function associated with the
 * edge in a given adjacent element
 *
 * @param edgeNo given edge
 * @param elemNo element the edge is associated with
 * @param  *mesh input mesh
 * return  index local index of edge in element
 *
 */
int findEdgeLocalIndex(int edgeNo, int elemNo, Mesh* mesh)
{
    Array<int> elem_edges, cor; 
    mesh->GetElementEdges(elemNo, elem_edges, cor);
    for (int j = 0; j < elem_edges.Size(); j++)
    {
      if(elem_edges[j] == edgeNo) return j;
    }
}
 
int findEdgeLocalIndexInFace(int edgeNo, int faceNo, Mesh* mesh)
{
    Array<int> face_edges, cor; 
    mesh->GetFaceEdges(faceNo, face_edges, cor);
    for (int j = 0; j < face_edges.Size(); j++)
    {
      if(face_edges[j] == edgeNo) return j;
    }
} 

/**
 * This method computes the flux term integral on the two faces
 * of a given element in the patch. 
 *
 * @param   edgeNo edge around which the patch is formed 
 * @param   elemNo given element for which the flux term integral is computed on two of its faces
 * @param      *fi all upward adjacent faces in the edge patch
 * @param       nf number of upward adjacent faces in the edge patch
 * @param    *mesh input mesh
 * @param        x FEM solution
 * @param *fespace finite element space
 * TODO for higher order, might want to include the flux term integral on all faces of the
 *      element rather than just the two faces in the edge patch. 
 *
 */
double computeFluxTermIntegral(int edgeNo, int elemNo, Vector ord_fi, Mesh* mesh,
                               GridFunction x, FiniteElementSpace* fespace)
{
  double integralVal = 0.0;
  Vector fluxFaces;
  getElemCavityFaces(elemNo, ord_fi, mesh, fluxFaces);
  int edgeIndex = findEdgeLocalIndex(edgeNo, elemNo, mesh); // get local index of the edge in elem for shape function  
  for(int ff = 0; ff < fluxFaces.Size(); ff++) // compute integral on each flux face separately and sum
  {
    int faceNo = fluxFaces[ff];
    integralVal += computeFluxFaceIntegral(faceNo, elemNo, edgeIndex, mesh, fespace, x);  
  }
  return integralVal;
}

/**
 * This method computes the bilinear form integral on a 
 * given element in the edge patch. 
 *
 * TODO comment  
 * 
 *
 *
 */
double computeBilinearFormIntegral(int edgeNo, int elemNo, Mesh* mesh, GridFunction x, 
                                   FiniteElementSpace* fespace, CurlCurlIntegrator* cci, 
                                   VectorFEMassIntegrator* mi)
{
  // get local edge index for shape function
  int edgeIndex = findEdgeLocalIndex(edgeNo, elemNo, mesh);
  
  // Assemble local element matrix for elemNo
  const FiniteElement* fe = fespace->GetFE(elemNo);
  ElementTransformation* eltr = mesh->GetElementTransformation(elemNo);
  int ndofs = fe->GetDof();
  DenseMatrix elmat(ndofs, ndofs); elmat = 0.0;

  DenseMatrix elmat_curl(ndofs, ndofs); elmat_curl = 0.0;
  cci->AssembleElementMatrix(*fe, *eltr, elmat_curl);
  elmat.Add(1.0, elmat_curl);
  
  DenseMatrix elmat_mass(ndofs, ndofs); elmat_mass = 0.0;
  mi->AssembleElementMatrix(*fe, *eltr, elmat_mass);
  elmat.Add(1.0, elmat_mass);

  // Get the global edge dof values of the elemNo
  Array<int> eldofs;
  fespace->GetElementDofs(elemNo, eldofs);
  Vector eldofs_vals(eldofs.Size()); 
  x.GetSubVector(eldofs, eldofs_vals); // TODO fix the negative dof value??
  cout << "Element Dofs for elemNo" << elemNo << ": "; eldofs.Print(cout, 6);
  cout << "Dof Values for elemNo " << elemNo << ": "; eldofs_vals.Print(cout, 6);

  // Multiply the vector with elmat
  Vector vals(eldofs.Size()); vals = 0.0;
  elmat.Mult(eldofs_vals,vals);
  cout << "Bilinear Form (1) elemNo " << elemNo << ": " ; vals.Print(cout, ndofs);
  if(eldofs[edgeIndex] < 0) { return -1 * vals(edgeIndex); } // TODO taking care of negative dof value
  return vals(edgeIndex); 
}

/**
 * This method computes the linear form on a given element in the edge patch.
 *
 * @param   edgeNo 
 * @param   elemNo
 * @param    *mesh 
 * @param *fespace
 */
double computeLinearFormIntegral(int edgeNo, int elemNo, Mesh* mesh, 
                                 FiniteElementSpace* fespace, VectorFEDomainLFIntegrator* dlfi)
{
   int edgeIndex = findEdgeLocalIndex(edgeNo, elemNo, mesh);
   const FiniteElement* fe = fespace->GetFE(elemNo);

   // Integration Points are handled inside AssembleRHSElementVect (2*fe->GetOrder())
   ElementTransformation* tr = mesh->GetElementTransformation(elemNo);
   Vector elvect(fe->GetDof()); elvect = 0.0;
   dlfi->AssembleRHSElementVect(*fe, *tr, elvect);
   Array<int> eldofs; fespace->GetElementDofs(elemNo, eldofs);  // TODO Taking care of negative dof indices
   if(eldofs[edgeIndex] < 0) { return -1 * elvect(edgeIndex); } // TODO Takinf care of negative dof indices
   
   return elvect(edgeIndex);
}

/**
 * This method assembles the mass matrix on the face
 * to solve for the theta values.
 *
 * @param  faceNo current face on which the matrix is to be assembled
 * @param g_edges the edges of the faceNo in the order corresponding to
 *                the order of the entries in RHS g vector
 */
 void computeFaceMassMatrix(int faceNo, Mesh* mesh,
                            FiniteElementSpace* fespace, DenseMatrix& facemat)
{
 
  // 1. CALCULATE THE DOF x DOF Mass matrix
  // Get the integration rule for the face
  FaceElementTransformations* ftr = mesh->GetFaceElementTransformations(faceNo);
  int order = 2 * ftr->Elem1->Order(); 
  const IntegrationRule *ir = &IntRules.Get(ftr->FaceGeom, order);

  // Get the finite element on the face
  const FiniteElement* fe = fespace->GetFaceElement(faceNo);
  int dof = fe->GetDof();
  int dim = ftr->Elem1->GetSpaceDim();
  DenseMatrix vshape(dof, dim); vshape = 0.0; // dof x dim (3 x 3)
  facemat.SetSize(dof);

  // test 
  Array<int> facedofs; fespace->GetFaceDofs(faceNo, facedofs);
  cout << " ND Face " << faceNo << " Dofs "; facedofs.Print(cout, facedofs.Size()); 

  // Calculate 3x3 face mass matrix 
  double w = 0.0;
  facemat = 0.0; 
  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    const IntegrationPoint &ip = ir->IntPoint(i);
    ftr->Face->SetIntPoint(&ip);
    fe->CalcVShape(*ftr->Face, vshape); 

    // TODO testing negative dof indices
    for(int ind = 0; ind < facedofs.Size(); ind++)
    {
      if(facedofs[ind] < 0)
      {
        for(int j = 0; j < dim; j++) 
          {vshape(ind,j) = -1*vshape(ind,j);}
      }
    }

    w = ip.weight * ftr->Face->Weight();
    AddMult_a_AAt(w, vshape, facemat);
  }
}

/**
 * This method evaluates the theta vector on a face as a linear combination
 * of the shape functions associated with the edges of the face and using
 * the corresponding theta parameters as the weights in the expansion.
 *
 * @param faceNo the face on which theta vector is being evaluated
 * @param elemNo the current upward adjacent element of the face
 * @param 
 */
void evaluateThetaVector(int faceNo, Vector theta_values, Mesh* mesh,
                         FiniteElementSpace* fespace, Vector& theta_vec)
{
  // Get the integration point for the face
  FaceElementTransformations* ftr = mesh->GetFaceElementTransformations(faceNo);
  Geometry::Type geo_type = mesh->GetFaceGeometryType(ftr->Face->ElementNo);
  const IntegrationPoint *center = &Geometries.GetCenter(geo_type); // center of face
  ftr->Face->SetIntPoint(center);

  // Get the finite element on the face
  const FiniteElement* fe = fespace->GetFaceElement(faceNo);
  int dof = fe->GetDof();
  int dim = ftr->Elem1->GetSpaceDim();
  DenseMatrix vshape(dof, dim); vshape = 0.0; // dof x dim (3 x 3)

  // evaluate shape functions on the face
  fe->CalcVShape(*ftr->Face, vshape); 

  // TODO test ND negative dofs
  Array<int> facedofs; fespace->GetFaceDofs(faceNo, facedofs);
  //cout << " ND Face " << faceNo << " Dofs "; facedofs.Print(cout, facedofs.Size()); 

  // Compute theta vector as a linear combination of the shape functions
  theta_vec = 0.0;
  for(int i = 0; i < theta_values.Size(); i++)
  {
    Vector shape;
    vshape.GetRow(i, shape);

    // TODO testing negative dof indices
    if(facedofs[i] < 0)
      shape.Neg();

    theta_vec.Add(theta_values(i), shape); 
  }
  cout << "Theta Vector for FaceNo " << faceNo << ": "; theta_vec.Print(cout, theta_vec.Size());  
  // Testing: compute normal to the face and then compute
  // dot product of normal with theta vector to see if they
  // are orthgonal.
  //Vector normal(dim);
  //CalcOrtho(ftr->Face->Jacobian(), normal);
  //cout << "Theta vector dot with face normal = " << theta_vec * normal << endl;
}

/**
 *
 * This method arranges edge patch entities in an order.
 * Upward adjacent elements and faces of the edge
 * are arranged in an order. However, we do not guarantee
 * counter clockwise or clockwise order. By convention,
 * the first face is the one shared by the last and the 
 * first element in the reordered list. This method works
 * only for interior edge patches currently. 
 *
 * TODO make this method more efficient (optimize)
 * TODO also include exterior edge patches in it
 *
 */
 void orderEdgePatchEntities(int edgeNo, int ne, int nf, const int* ei, const int* fi,
                               Mesh* mesh, Vector& ord_ei, Vector& ord_fi)
 {
   // order interior edge patch entities
   if(1 !=  edgeIsExterior(edgeNo, nf, fi, mesh))  // FOR INTERIOR EDGE PATCHES
   {
     // order elements
     ord_ei.SetSize(ne);
     for(int i = 0; i < ne; i++) { ord_ei(i) = ei[i]; } // copy current ei to ord_ei

     // loop over all elements in ei except last
     for(int elNo = 0; elNo < ne-1; elNo++)
     {
       Array<int> elm_faces, elm_faces_cor;
       mesh->GetElementFaces(ord_ei(elNo), elm_faces, elm_faces_cor); //get downward faces of the element
       // another loop over elements (excluding elNo) to check which elements in ei share a face with elNo. (expecting only 2 such elements at max)
       for(int cmp = elNo+1; cmp < ne; cmp++)
       {
         Array<int> cmp_faces, cmp_faces_cor;
         mesh->GetElementFaces(ord_ei(cmp), cmp_faces, cmp_faces_cor);

         // compare the faces of elNo with the faces of cmp
         // if they both share a face, modify ord_ei to place them next to each other
         int sharedFaces = checkConnectivity(elm_faces, cmp_faces);
         if(sharedFaces == 1)
         {
           //cout << "Element " << ord_ei(elNo) << " and Element " << ord_ei(cmp) << " share faces " << endl;
           int temp = ord_ei(elNo+1);
           ord_ei(elNo+1) = ord_ei[cmp];
           ord_ei[cmp] = temp;
           //ord_ei.Print(cout, ord_ei.Size());
           break;
         }
       }
     }//------------------ELEMENTS ORDERED-----------------------//

     // order faces 
     ord_fi.SetSize(nf);
     // loop over all elements except last
     for(int elNo = 0; elNo < ne-1; elNo++)
     {
       Array<int> elm_faces, elm_faces_cor;
       mesh->GetElementFaces(ord_ei(elNo), elm_faces, elm_faces_cor);
 
       // get faces of the next element
       Array<int> elmNext_faces, elmNext_faces_cor;
       mesh->GetElementFaces(ord_ei(elNo+1), elmNext_faces, elmNext_faces_cor);

       // get the shared face
       int sharedFace;
       int breakNext = 0;
       for(int i = 0; i < elm_faces.Size(); i++)
       {
         for(int j = 0; j < elm_faces.Size(); j++)
         {
           if(elm_faces[i] == elmNext_faces[j]) 
           {
             sharedFace = elm_faces[i];
             breakNext++;
             break;       
           }
         }
         if(breakNext == 1) break;
       }
       ord_fi(elNo+1)= sharedFace; // set the shared face in ord_fi 
       //cout << "ord_fi "; ord_fi.Print(cout, ord_fi.Size()); cout <<endl;
     }

     // set the first face
     // check which face in fi is not in the ord_fi,
     // and set that as the last face in ord_fi
     for(int i = 0; i < nf; i++)
     {
       int count = 0;
       for(int j = 1; j < nf; j++)
       {
         if(fi[i] != ord_fi(j))
         {
           count++;
           if(count == nf-1)
           {
             ord_fi(0) = fi[i];
             break;
           }
         } 
       }
     }//--------------------------------FACES ORDERED-----------// 
  } // interior edge patch entities ordered
  else // FOR EXTERIOR EDGE PATCHES
  {
    // copy fi and ei into vector fi and ei
    Vector temp_fi, temp_ei;
    temp_fi.SetSize(nf);
    temp_ei.SetSize(ne);    
    for(int i = 0; i < nf; i++) { temp_fi(i) = fi[i]; } 
    for(int i = 0; i < ne; i++) { temp_ei(i) = ei[i]; } 
    //cout << "fi "; temp_fi.Print(cout, nf);
    //cout << "ei "; temp_ei.Print(cout, ne);



    // Start ordering of the entities
    ord_ei.SetSize(ne); 
    ord_fi.SetSize(nf);
    int findex = 0;
    int eindex = 0;

    // take the first face which is on the boundary and assign it to the first
    // face face in the ord_fi list. 
    for(int i = 0; i < nf; i++)
    {
      if( !mesh->FaceIsInterior(fi[i]) )
      {
        ord_fi(findex) = fi[i]; break;
      }
    }
    findex++;

    // then take the element attached to that face (should only be one element)
    // and assign it to the first element in the ord_ei list. 
    FaceElementTransformations* ftr = mesh->GetFaceElementTransformations(ord_fi(findex-1));
    if(ftr->Elem1No != -1) { ord_ei(eindex) = ftr->Elem1No; }
    else { ord_ei(eindex) = ftr->Elem2No; }
    eindex++;
    
    while(findex < nf)
    {
      Vector elem_faces;
      int elemNo = ord_ei(eindex-1);      
      getElemCavityFaces(elemNo, temp_fi, mesh, elem_faces);
      for(int i = 0; i < elem_faces.Size(); i++)
      {
        if(elem_faces(i) != ord_fi(findex-1)) { ord_fi(findex) = elem_faces(i); }
      } 

      if ( eindex < ne )
      {
        FaceElementTransformations* ftr = mesh->GetFaceElementTransformations(ord_fi(findex));
        if(ftr->Elem1No != ord_ei(eindex-1)) { ord_ei(eindex) = ftr->Elem1No; }
        else { ord_ei(eindex) = ftr->Elem2No; }
        eindex++;
      }

      findex++;
    } 

    
    //cout << "ord_fi "; ord_fi.Print(cout, nf);
    //cout << "ord_ei "; ord_ei.Print(cout, ne);

  } // boundary edge patch entities ordered
}

/**
 * Demkowicz approach
 * Assemble Interior LHS singular matrices based 
 * on the description given in Demkowicz book
 * 
 *
 *
 *
 */
 void assembleInteriorLHS(int edgeNo, Vector ord_ei, Vector ord_fi,
                     Mesh* mesh, DenseMatrix& M)
 {
   M = 0.0;
   int ne = ord_ei.Size();
   int nf = ord_fi.Size(); 

   // loop over columns (faces)
   for(int cc = 0; cc < nf; cc++)
   {
     int faceNo = ord_fi[cc];
     FaceElementTransformations* ftr = mesh->GetFaceElementTransformations(faceNo);
     int elem1No = ftr->Elem1No;
     int elem2No = ftr->Elem2No;
   
     // find the position of the two upward adjacent elements in ord_ei
     int elem1pos; int elem2pos;
     for(int i = 0; i < ne; i++)
     {
       if(elem1No == ord_ei[i])
       {
         elem1pos = i;
       }

       if(elem2No == ord_ei[i])
       {
         elem2pos = i;
       }
     }
    
     // set the entries
     if(cc == 0)
     {
       if(elem1pos < elem2pos)
       {
         M(elem1pos, cc) = 1.0;
         M(elem2pos, cc) = -1.0;
       }
       else
       {
         M(elem1pos, cc) = -1.0;
         M(elem2pos, cc) = 1.0;
       }
     }
     else
     {
       if(elem1pos < elem2pos)
       {
         M(elem1pos, cc) = -1.0;
         M(elem2pos, cc) = 1.0;
       }
       else
       {
         M(elem1pos, cc) = 1.0;
         M(elem2pos, cc) = -1.0;
       }
     }
   }
}


/**
 * Demkowicz approach
 * Assemble Exterior LHS matrices based 
 * on the description given in Demkowicz book
 * 
 * TODO Natural BC - Natural BC lhs 
 * TODO Natural BC - Dirichlet BC lhs 
 *
 */
 void assembleExteriorLHS(int edgeNo, Vector ord_ei, Vector ord_fi,
                     Mesh* mesh, DenseMatrix& M)
 {
   M = 0.0;
  
   int ne = ord_ei.Size();
   int nf = ord_fi.Size(); 
   // loop over columns (faces)
   for(int cc = 0; cc < nf; cc++)
   {
     int faceNo = ord_fi[cc];
     FaceElementTransformations* ftr = mesh->GetFaceElementTransformations(faceNo);
     int elem1No = ftr->Elem1No;
     int elem2No = ftr->Elem2No;
   
     // find the position of the two upward adjacent elements in ord_ei
     int elem1pos; int elem2pos;
     for(int i = 0; i < ne; i++)
     {
       if(elem1No == ord_ei[i])
       {
         elem1pos = i;
       }

       if(elem2No == ord_ei[i])
       {
         elem2pos = i;
       }
     }
    
     // set the entries
     if(cc == 0 || cc == nf-1)
     {
         M(elem1pos+nf, cc) = 1.0;
         M(cc, elem1pos+nf) = 1.0; // symmetric
     }
     else
     {
       if(elem1pos < elem2pos)
       {
         M(elem1pos+nf, cc) = -1.0;
         M(elem2pos+nf, cc) = 1.0;

         M(cc, elem1pos+nf) = -1.0; // symmetric entries
         M(cc, elem2pos+nf) = 1.0;
       }
       else
       {
         M(elem1pos+nf, cc) = 1.0;
         M(elem2pos+nf, cc) = -1.0;

         M(cc, elem1pos+nf) = 1.0; // symmetric entries
         M(cc, elem2pos+nf) = -1.0;
       }
     }

     // set the 2*m on the diagonals (m =1 now) 
     // TODO use size of the face nf as m1, m2, m3, ... 
     M(cc,cc) = 2.0;
   } // minimized matrix assembled 
}

/*
 *
 *
 *
 */
void computeLambda_BVP(int elemNo, FiniteElementSpace* fespace, Mesh* mesh,
                       std::vector<vector<double>> THETA_COEFF, GridFunction x,
                       Vector& lambda_k)
{
  cout << "COMPUTE LAMBDA BVP FOR ELEMNO " << elemNo << endl;

  const FiniteElement* fel = fespace->GetFE(elemNo);
  ElementTransformation* eltr = mesh->GetElementTransformation(elemNo);
  int neldofs = fel->GetDof();
  int dim = fel->GetDim();
  int order = fel->GetOrder(); 
  lambda_k.SetSize(neldofs); 
  lambda_k = 0.0;

  // get the four downward faces of the element
  Array<int> elem_fcs, faces_or;
  mesh->GetElementFaces(elemNo, elem_fcs, faces_or); 
  int nfaces = elem_fcs.Size();

  Vector testfaceint(neldofs); testfaceint = 0.0;
  // assemble the [ndofsx1] lambda vector
  for(int ii = 0; ii < nfaces; ii++)
  {
    int faceNo = elem_fcs[ii];
    FaceElementTransformations* ftr = mesh->GetFaceElementTransformations(faceNo);
    int elemNo1 = ftr->Elem1No;
    int elemNo2 = ftr->Elem2No; 

    const FiniteElement* facefe = fespace->GetFaceElement(faceNo);
    int nfdofs = facefe->GetDof();
    DenseMatrix facevshape(nfdofs, dim); facevshape = 0.0; // dof x dim (3 x 3)

    Array<int> facedofs; fespace->GetFaceDofs(faceNo, facedofs);
    cout << " ND Face " << faceNo << " Dofs "; facedofs.Print(cout, facedofs.Size()); 

    // get theta coefficients on each face
    Vector theta_coeff(dim);
    for(int i = 0; i < dim; i++) { theta_coeff(i) = THETA_COEFF[faceNo][i]; } 

    // compute integral on the face
    Vector faceint(neldofs); faceint = 0.0;
    const IntegrationRule *ir = &IntRules.Get(ftr->FaceGeom, 2*order);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
      const IntegrationPoint &ip = ir->IntPoint(i);
      ftr->Face->SetIntPoint(&ip);
      IntegrationPoint eip1;
      ftr->Loc1.Transform(ip, eip1);
      ftr->Elem1->SetIntPoint(&eip1);
      IntegrationPoint eip2;
      ftr->Loc2.Transform(ip, eip2);
      if(mesh->FaceIsInterior(faceNo))
      { 
        ftr->Elem2->SetIntPoint(&eip2);
      }

      // evaluate theta vector using ip
      Vector theta_vec(dim); theta_vec = 0.0;
      facefe->CalcVShape(*ftr->Face, facevshape);
      for(int i = 0; i < theta_coeff.Size(); i++)
      {
        Vector shape;
        facevshape.GetRow(i, shape);
      
        if(facedofs[i] < 0)
          shape.Neg();

        theta_vec.Add(theta_coeff(i), shape); 
      }
      if(elemNo == elemNo2) { theta_vec.Neg(); }
      cout << "Theta Vector for FaceNo " << faceNo << ": "; theta_vec.Print(cout, theta_vec.Size()); 

      // evalute flux vector tk using ip
      // 1. compute outward unit normal on the face
      Vector normal(dim); normal = 0.0;
      computeFaceOutwardNormal(faceNo, elemNo, normal, mesh);
      normalizeVector(normal); 
      cout << "Normal for FaceNo " << faceNo << ": "; normal.Print(cout, theta_vec.Size()); 

      // 2. Compute curl and tk
      Vector t_k(dim); t_k = 0.0;
      if(mesh->FaceIsInterior(faceNo))
      {
        normal /= 2.0;
        Vector curl1, curl2; 
        x.GetCurl(*ftr->Elem1, curl1);         
        x.GetCurl(*ftr->Elem2, curl2);         
        Vector sumCurl(dim); sumCurl = 0.0;
        sumCurl.Add(1.0, curl1);
        sumCurl.Add(1.0, curl2);
        cross3(normal, sumCurl, t_k); // get t_k
        cout << "t_k for FaceNo " << faceNo << " and Elem No " << elemNo << " ";
        t_k.Print(cout, dim);
      }
      else // Dirichlet face on the boundary
      {
        Vector curl1; 
        x.GetCurl(*ftr->Elem1, curl1);         
        cross3(normal, curl1, t_k); // get t_k
        cout << "(Boundary Face) t_k for FaceNo " << faceNo << " and Elem No " << elemNo << endl;
        t_k.Print(cout, t_k.Size());
      }

      // add theta vector and flux vector
      Vector theta_plus_tk(dim); theta_plus_tk = 0.0;
      theta_plus_tk.Add(1.0, theta_vec);
      theta_plus_tk.Add(1.0, t_k);
      cout << "Theta Vector plus flux vector for FaceNo " << faceNo << ": "; theta_plus_tk.Print(cout, theta_vec.Size()); 

      // compute integral on face
      if(elemNo == elemNo1)
      {
        const FiniteElement* fel1 = fespace->GetFE(elemNo1);
        DenseMatrix elemvshape(neldofs, dim); // dof x dim
        fel1->CalcVShape(*ftr->Elem1, elemvshape); 
        cout << "ElemVShape Matrix " << elemNo << endl; printDenseMatrix(elemvshape); cout << endl;       
 
        // negating to take care of negative dofs in Nedelec spaces
        Array<int> eldofs; fespace->GetElementDofs(elemNo, eldofs); 
        cout << "Eldfos " << elemNo << ": "; eldofs.Print(cout, eldofs.Size());
        for(int ind = 0; ind < eldofs.Size(); ind++)
        {
          if(eldofs[ind] < 0)
          {
            for(int j = 0; j < dim; j++) 
              {elemvshape(ind,j) = -1*elemvshape(ind,j);}
          }
        }

        Vector w_theta_plus_tk(dim); 
        w_theta_plus_tk.Set(ip.weight * ftr->Face->Weight(), theta_plus_tk);
        Vector temp(neldofs); temp = 0.0;
        elemvshape.Mult(w_theta_plus_tk, temp); 
        lambda_k.Add(1.0, temp); 
        testfaceint.Add(1.0, temp);   // test
      }
      else
      {
        const FiniteElement* fel2 = fespace->GetFE(elemNo2);
        DenseMatrix elemvshape(neldofs, dim); // dof x dim
        fel2->CalcVShape(*ftr->Elem2, elemvshape); 
        cout << "ElemVShape Matrix " << elemNo << endl; printDenseMatrix(elemvshape); cout << endl;       

        // negating to take care of negative dofs in Nedelec spaces
        Array<int> eldofs; fespace->GetElementDofs(elemNo, eldofs);
        cout << "Eldfos " << elemNo << ": "; eldofs.Print(cout, eldofs.Size());
        for(int ind = 0; ind < eldofs.Size(); ind++)
        {
          if(eldofs[ind] < 0)
          {
            for(int j = 0; j < dim; j++) 
              {elemvshape(ind,j) = -1*elemvshape(ind,j);}
          }
        }

        Vector w_theta_plus_tk(dim); 
        w_theta_plus_tk.Set(ip.weight * ftr->Face->Weight(), theta_plus_tk);
        Vector temp(neldofs); temp = 0.0;
        elemvshape.Mult(w_theta_plus_tk, temp); 
        lambda_k.Add(1.0, temp); 
        testfaceint.Add(1.0, temp);   // test
      }
/*
      cout << "IP " << i << " of " << ir->GetNPoints() << endl;
      cout << "weight " << ip.weight << " jdet " << eltr->Weight() << endl;
      cout << "w " << ip.weight * eltr->Weight() << endl;
      cout << "Theta_plus_tk "; theta_plus_tk.Print(cout,dim); cout << endl;
      cout << "CalcVShape " << endl; printDenseMatrix(vshape); cout << endl;
      cout << "Temp " << endl; temp.Print(cout, 1); cout << endl;
      cout << "Lambda vector " << endl; lambda_k.Print(cout, 1); cout << endl;
*/ 
    }
    cout << "lambda face integral " << faceNo << ": "; testfaceint.Print(cout, neldofs); 
  }
}

/*
 *
 *
 */
void computeBilinearForm_BVP(int elemNo, Mesh* mesh, GridFunction x, 
                             FiniteElementSpace* fespace, Vector& b_k)
{
  const FiniteElement* fe = fespace->GetFE(elemNo);
  ElementTransformation* eltr = mesh->GetElementTransformation(elemNo);
  int ndofs = fe->GetDof();
  int dim = fe->GetDim();
  int order = fe->GetOrder(); 
  b_k.SetSize(ndofs); 
  b_k = 0.0;

  // 1. Compute CurlCurl Integration 
  const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2*order-2); 
  double w;
  Vector curlcurl(ndofs); curlcurl = 0.0;
  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    const IntegrationPoint &ip = ir->IntPoint(i);
    eltr->SetIntPoint(&ip);
    Vector curl_vec(dim);
    x.GetCurl(*eltr, curl_vec); // get Curl of Solution in physical space on IP
   
    DenseMatrix curlShape(ndofs,dim), curlShape_dFt(ndofs,dim);
    fe->CalcCurlShape(ip, curlShape); // getCurlShape in reference space on IP 
    MultABt(curlShape, eltr->Jacobian(), curlShape_dFt); // transform to physical space
     
    w = ip.weight; // eltr->Weight(); // mapping for curlcurl integration /TODO //ERROR FOUND HERE

    Vector temp(ndofs); temp = 0.0;
    curlShape_dFt.Mult(curl_vec, temp);
    temp *= w;

    curlcurl.Add(1.0, temp);
  }
       
  // 2. Compute VectorMass Integration 
  ir = &IntRules.Get(fe->GetGeomType(), 2*order);
  Vector femass(ndofs); femass = 0.0;
  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    const IntegrationPoint &ip = ir->IntPoint(i);

    Vector vval(dim);
    x.GetVectorValue(elemNo, ip, vval); // get E_h

    eltr->SetIntPoint(&ip);
    DenseMatrix vshape(ndofs, dim);
    fe->CalcVShape(*eltr, vshape);  // get vshape in physical space on IP 

    w = ip.weight * eltr->Weight();

    Vector temp(ndofs); temp = 0.0;
    vshape.Mult(vval, temp);
    temp *= w;

    femass.Add(1.0, temp);
  }

  // 3. Get result
  b_k.Add(1.0, curlcurl);
  b_k.Add(1.0, femass);

  // Negative ND Dofs
  Array<int> eldofs; 
  fespace->GetElementDofs(elemNo, eldofs);
  for(int i = 0; i < eldofs.Size(); i++)
  {
    if(eldofs[i] < 0)
     b_k(i) =  -1 * b_k(i); // TODO taking care of negative dof value
  }
  
}

/**
 *
 *
 *
 */
void applyDBC(int elemNo, Mesh* mesh, FiniteElementSpace* fespace, Table edge_face)
{
  Array<int> eldofs;
  fespace->GetElementDofs(elemNo, eldofs);
  cout << "Element " << elemNo << " Dofs "; eldofs.Print(cout, eldofs.Size());
  Array<int> elem_edges, edges_cor;
  mesh->GetElementEdges(elemNo, elem_edges, edges_cor);
  cout << "Element " << elemNo << " Edges "; elem_edges.Print(cout, elem_edges.Size());
  cout << "Element " << elemNo << " Edge Orientations "; edges_cor.Print(cout, elem_edges.Size()); cout << endl;

  // Face Dofs
  Array<int> elem_faces, faces_cor;
  mesh->GetElementFaces(elemNo, elem_faces, faces_cor);
  for(int i = 0; i < elem_faces.Size(); i++)
  {
    int faceNo = elem_faces[i];
    if(!mesh->FaceIsInterior(faceNo))
    {
      Array<int> facedofs;
      fespace->GetFaceDofs(faceNo, facedofs);
      cout << "Exterior Face " << faceNo << " Dofs "; facedofs.Print(cout, facedofs.Size());
      Array<int> faceintdofs;
      fespace->GetFaceInteriorDofs(faceNo, faceintdofs);
      cout << "Exterior Face " << faceNo << " Interior Dofs "; faceintdofs.Print(cout, faceintdofs.Size());
      Array<int> face_edges, faceedges_cor;
      mesh->GetFaceEdges(faceNo, face_edges, faceedges_cor);
      cout << "Exterior Face " << faceNo << " Edges "; face_edges.Print(cout, face_edges.Size()); cout << endl;
      for(int ee = 0; ee < face_edges.Size(); ee++)
      {
        int edgeNo = face_edges[ee];
        Array<int> edgedofs;
        fespace->GetEdgeDofs(edgeNo, edgedofs);
        cout << "EdgeNo " << edgeNo << " Dofs "; edgedofs.Print(cout, edgedofs.Size());
      }
    }
    else
    {
      cout << "Interior Face " << faceNo << endl;
    }
  }


  // Get the local indices of boundary dofs in eldofs
  std::vector<int> local_bdr_dofs;
  // loop over edges to get edge dofs on bdry
  for(int ee = 0; ee < elem_edges.Size(); ee++) 
  {
    int edgeNo = elem_edges[ee];
    int nf = edge_face.RowSize(edgeNo);
    const int* fi = edge_face.GetRow(edgeNo);
    cout << "Adjacent Faces of EdgeNo " << edgeNo << " : "; for(int i = 0; i < nf; i++) { cout << fi[i] << " "; } cout << endl;

    if (1 == edgeIsExterior(edgeNo, nf, fi, mesh))
    {
      cout << "EdgeNo " << edgeNo << " is on Boundary " << endl;
      Array<int> edgedofs;
      fespace->GetEdgeDofs(edgeNo, edgedofs); // get edge dofs

      int index = 0;
      for(int dd = 0; dd < edgedofs.Size(); dd++) // loop edge dofs
      {
        for(int i = 0; i < eldofs.Size(); i++)
        {
          if(eldofs[i] == edgedofs[dd] || eldofs[i] == (-1 - edgedofs[dd])) 
            {local_bdr_dofs.push_back(i); break;}
          else index++;
        }
      }
    }    
  }

  // loop over bdry faces to get face dofs
  for(int ff = 0; ff < elem_faces.Size(); ff++) 
  {
    int faceNo = elem_faces[ff];
    if(!mesh->FaceIsInterior(faceNo)) // if bdry face
    {
      Array<int> faceintdofs;
      fespace->GetFaceInteriorDofs(faceNo, faceintdofs); // get interior face dofs
      
      int index = 0;
      for(int dd = 0; dd < faceintdofs.Size(); dd++) // loop interior dofs
      {
        for(int i = 0; i < eldofs.Size(); i++)
        {
          if(eldofs[i] == faceintdofs[dd]) 
            {local_bdr_dofs.push_back(i); break;}
          else index++;
        }
      }
    }
  }

  cout << "Dirichlet Boundary Dofs "; for(int i = 0; i < local_bdr_dofs.size(); i++) { cout << local_bdr_dofs[i] << " "; }
        
    
   



  
  cout << " --------------- " << endl;
}


/**
 * TODO Right now this function imposes a Dirichlet BC on all boundary
 * faces of an element. Need to change it so, that Dirichlet BC is
 * imposed only on Dirichlet boundary faces and not on Natural or Mixed
 * boundary faces.
 *
 */
void getEssentialElementTrueDofs(int elemNo, const Array<int> &bdr_attr_is_ess,
                                 Array<int> &ess_elem_tdofs_list,
                                 Array<int> &uness_elem_tdofs_list,
                                 Mesh* mesh, FiniteElementSpace* fespace)
{
  // Get Element Dofs
  Array<int> elem_dofs;
  fespace->GetElementDofs(elemNo, elem_dofs);
  cout << "Initial Dof List "; elem_dofs.Print(cout, elem_dofs.Size());
  for(int i = 0; i < elem_dofs.Size(); i++) // make all elem dofs positive
  { 
    if (elem_dofs[i] < 0)
      elem_dofs[i] = -1 - elem_dofs[i];
  }
  cout << "Final Dof List "; elem_dofs.Print(cout, elem_dofs.Size());
  
  const FiniteElement* fe = fespace->GetFE(elemNo);
  Array<int> ess_elem_tdofs;
  ess_elem_tdofs.SetSize(fe->GetDof());
  ess_elem_tdofs = 0; // marker list

  // Get Element Faces
  Array<int> elem_faces, fo;
  mesh->GetElementFaces(elemNo, elem_faces, fo);
  
  Array<int> dofs;
  for(int ff = 0; ff < elem_faces.Size(); ff++) // bdr element loop
  {
    int faceNo = elem_faces[ff];
    if(!mesh->FaceIsInterior(faceNo)) // boundary faces
    {
      //if(bdr_attr_is_ess[mesh->GetAttribute(faceNo)-1]) // -1 is questionable?
      {
        fespace->GetFaceDofs(faceNo, dofs);
        cout << "Initial Face " << faceNo << " Dof List "; dofs.Print(cout, dofs.Size());
        for(int i = 0; i < dofs.Size(); i++) // make all face dofs positive
        { 
          if (dofs[i] < 0)
            dofs[i] = -1 - dofs[i];
        }
        cout << "Final Face " << faceNo << " Dof List "; dofs.Print(cout, dofs.Size());

        for(int dd = 0; dd < dofs.Size(); dd++) // mark dofs
        {
          int k = dofs[dd];
          for(int i = 0; i < elem_dofs.Size(); i++)
          {
            if(k == elem_dofs[i])
              {ess_elem_tdofs[i] = -1; break;}
          }
        }
      }
    }
  }
  cout << "MARKED LIST "; ess_elem_tdofs.Print(cout, ess_elem_tdofs.Size()); cout << endl;

  // Marker to List (ess_elem_tdofs -> ess_elem_tdofs_list)
  int num_marked = 0;
  for(int i = 0; i < ess_elem_tdofs.Size(); i++)
  {
  	if(ess_elem_tdofs[i]) {num_marked++; }
  }
  ess_elem_tdofs_list.SetSize(0);
  ess_elem_tdofs_list.Reserve(num_marked);
  uness_elem_tdofs_list.SetSize(0); // test
  uness_elem_tdofs_list.Reserve(elem_dofs.Size() - num_marked); // test
 
  for(int i = 0; i < ess_elem_tdofs.Size(); i++)
  {
  	if(ess_elem_tdofs[i]) {ess_elem_tdofs_list.Append(i);}
        else {uness_elem_tdofs_list.Append(i); }
  }
  cout << " --------------------------------------------------- " << endl; cout << endl;
}
 

/**
 * Inputs: Matrix A, Vector X, Vector B, essential dofs, not essential dofs.
 * Output: reduced matrix A, reduced rhs B.
 *
 */
void eliminateBCs(DenseMatrix& A, Vector& X, Vector& B, Array<int> ess_tdofs_list, Array<int> uness_tdofs_list,
                  DenseMatrix& Anew, Vector& Bnew)
{
  // 1. Remove Rows of A corresponding to ess_tdofs_list
  //    OR Assemble new A copying data from A.
  int num_ess_dofs = ess_tdofs_list.Size();
  int num_other_dofs = uness_tdofs_list.Size();
  
  Anew.SetSize(num_other_dofs);
  for(int rr = 0; rr < num_other_dofs; rr++)
  {
    int i = uness_tdofs_list[rr];
    for(int cc = 0; cc < num_other_dofs; cc++)
    {
      int j = uness_tdofs_list[cc];
      Anew(rr,cc) = A(i,j);
    }    
  }
  cout << "Anew " << endl;
  printDenseMatrix(Anew);
  

  // 2. Assemble new B
  Bnew.SetSize(num_other_dofs);
  for(int i = 0; i < num_other_dofs; i++)
  {
    Bnew(i) = B(uness_tdofs_list[i]);
  }
  cout << "Bnew " << endl; Bnew.Print(cout, num_other_dofs); 

  
  // 3. Subtract from B
  // B -= Ae*Xe 
  DenseMatrix Ae(num_other_dofs, num_ess_dofs);
  for(int rr = 0; rr < num_other_dofs; rr++)
  {
    int i = uness_tdofs_list[rr];
    for(int cc = 0; cc < num_ess_dofs; cc++)
    {
      int j = ess_tdofs_list[cc];
      Ae(rr,cc) = A(i,j);
    }    
  }
  cout << "Ae " << endl;
  printDenseMatrix(Anew);

  
  // this contains the known DBC dof values (e = 0);
  Vector Xe(num_ess_dofs); 
  for(int i = 0; i < num_ess_dofs; i++)
  {
    Xe(i) = X(ess_tdofs_list[i]);
  }
  cout << "Xe " << endl; Xe.Print(cout, num_ess_dofs); 
  
  Ae.AddMult_a (-1.0, Xe, Bnew);

}             



/**
 *  Compute exact error in L2 Norm per element
 *
 */
double computeElementExactL2Error(int elemNo, VectorCoefficient &exsol, Mesh* mesh,
                                  FiniteElementSpace* fespace, GridFunction x)
{
  double error = 0.0;
  const FiniteElement *fe = fespace->GetFE(elemNo);
  int intorder = 2*fe->GetOrder() + 1;
  ElementTransformation* eltr = fespace->GetElementTransformation(elemNo);
  const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);
  DenseMatrix vals, exact_vals;
  Vector loc_errs;

  x.GetVectorValues(*eltr, *ir, vals);
  exsol.Eval(exact_vals, *eltr, *ir);
  vals -= exact_vals;
  loc_errs.SetSize(vals.Width());
  vals.Norm2(loc_errs); 
 
  for (int j = 0; j < ir->GetNPoints(); j++)
  {
    const IntegrationPoint &ip = ir->IntPoint(j);
    eltr->SetIntPoint(&ip);
    error += ip.weight * eltr->Weight() * (loc_errs(j) * loc_errs(j));
  }
  if (error < 0.0)
  {
    return -sqrt(-error);
  }
  return sqrt(error);
}

/*
 *
 */
double computeElementImplicitL2Error(int elemNo, Vector error_coeff, Mesh* mesh,
                                  FiniteElementSpace* fespace)
{
  double error = 0.0;
  const FiniteElement *fe = fespace->GetFE(elemNo);
  int intorder = 2*fe->GetOrder() + 1;
  ElementTransformation* eltr = fespace->GetElementTransformation(elemNo);
  Array<int> eldofs; fespace->GetElementDofs(elemNo, eldofs);
  const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);
  DenseMatrix vshape(fe->GetDof(), fe->GetDim());
  //DenseMatrix curlvshape(fe->GetDof(), fe->GetDim()); // curl energy norm
  Vector err_func(fe->GetDim());
 
  for (int j = 0; j < ir->GetNPoints(); j++)
  {
    const IntegrationPoint &ip = ir->IntPoint(j);
    eltr->SetIntPoint(&ip);

    fe->CalcVShape(*eltr, vshape); 
    // TODO Calc Curl shape here and compute energy norm
    for(int i = 0; i < eldofs.Size(); i++)
    {
      if(eldofs[i] < 0)
      {
        for(int j = 0; j < dim; j++)
        {
          vshape(i,j) = -vshape(i,j);
        }
      }
    }
    cout << "MATRIX SIZE " << vshape.Height() << " x " << vshape.Width() << endl;
    cout << "Vector size " << error_coeff.Size() << endl;
    vshape.MultTranspose(error_coeff, err_func);  

    double err_err = err_func * err_func;
    error += ip.weight * eltr->Weight() * err_err;
  }
  if (error < 0.0)
  {
    return -sqrt(-error);
  }
  return sqrt(error);
}


// ******* TESTING METHODS ******** //
//----------------------------------//
// TESTING METHOD - This method will print a matrix of type Densmatrix
void printDenseMatrix(DenseMatrix mat)
{
  for(int r = 0; r < mat.Height(); r++)  
  {
    for(int c = 0; c < mat.Width(); c++)
    {
      cout << mat(r,c) << " ";
    }
    cout << endl;
  }
}
/////////////////////////////////////////////////END PROGRAM//////////////////////////////////////////////////////////////////////
