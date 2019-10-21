//                                MFEM modified from Example 10 and 16
//
// Compile with: make imMHDp
//
// Description:  It solves a time dependent resistive MHD problem 
//               There three versions:
//               1. explicit scheme
//               2. implicit scheme using a very simple linear preconditioner
//               3. implicit scheme using physcis-based preconditioner
// Author: QT

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "BoundaryGradIntegrator.hpp"
#include "imResistiveMHDOperatorp.hpp"
#include "PCSolver.hpp"
#include <memory>
#include <iostream>
#include <fstream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

double alpha; //a global value of magnetude for the pertubation
double Lx;  //size of x domain
double lambda;
double resiG;

//initial condition
double InitialPhi(const Vector &x)
{
    return 0.0;
}

double InitialW(const Vector &x)
{
    return 0.0;
}

double InitialJ(const Vector &x)
{
   return -M_PI*M_PI*(1.0+4.0/Lx/Lx)*alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double InitialPsi(const Vector &x)
{
   return -x(1)+alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}


double BackPsi(const Vector &x)
{
   //this is the background psi (for post-processing/plotting only)
   return -x(1);
}

double InitialJ2(const Vector &x)
{
   return lambda/pow(cosh(lambda*(x(1)-.5)),2)
       -M_PI*M_PI*(1.0+4.0/Lx/Lx)*alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double InitialPsi2(const Vector &x)
{
   return log(cosh(lambda*(x(1)-.5)))/lambda
       +alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double BackPsi2(const Vector &x)
{
   return log(cosh(lambda*(x(1)-.5)))/lambda;
}

double E0rhs(const Vector &x)
{
   //for icase 2 only, there is a rhs
   return resiG*lambda/pow(cosh(lambda*(x(1)-.5)),2);
}

double InitialJ3(const Vector &x)
{
   double ep=.2;
   return (ep*ep-1.)/lambda/pow(cosh(x(1)/lambda) +ep*cos(x(0)/lambda), 2)
        -M_PI*M_PI*1.25*alpha*cos(.5*M_PI*x(1))*cos(M_PI*x(0));
}

double InitialPsi3(const Vector &x)
{
   double ep=.2;
   return -lambda*log( cosh(x(1)/lambda) +ep*cos(x(0)/lambda) )
          +alpha*cos(M_PI*.5*x(1))*cos(M_PI*x(0));
}

double BackPsi3(const Vector &x)
{
   double ep=.2;
   return -lambda*log( cosh(x(1)/lambda) +ep*cos(x(0)/lambda) );
}

double E0rhs3(const Vector &x)
{
   double ep=.2;
   return resiG*(ep*ep-1.)/lambda/pow(cosh(x(1)/lambda) +ep*cos(x(0)/lambda), 2);
}

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   //++++Parse command-line options.
   const char *mesh_file = "./Meshes/xperiodic-square.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 2;
   int ode_solver_type = 2;
   double t_final = 5.0;
   double dt = 0.0001;
   double visc = 1e-3;
   double resi = 1e-3;
   bool visit = false;
   bool use_petsc = false;
   bool use_factory = false;
   const char *petscrc_file = "";
   int icase = 1;
   alpha = 0.001; 
   Lx=3.0;
   lambda=5.0;

   bool visualization = true;
   int vis_steps = 10;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refineP",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - Brailovskaya,\n\t"
                  "            3 - L-stable SDIRK23, 4 - L-stable SDIRK33,\n\t"
                  "            22 - Implicit Midpoint, 23 - SDIRK23, 24 - SDIRK34.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&icase, "-i", "--icase",
                  "Icase: 1 - wave propagation; 2 - Tearing mode.");
   args.AddOption(&visc, "-visc", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&resi, "-resi", "--resistivity",
                  "Resistivity coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the nonlinear system.");
   args.AddOption(&use_factory, "-shell", "--shell", "-no-shell",
                  "--no-shell",
                  "Use user-defined preconditioner factory (PCSHELL).");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");

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
   if (icase==2)
   {
      resiG=resi;
   }
   else if (icase==3)
   {
      lambda=.5/M_PI;
      resiG=resi;
   }
   else if (icase==1)
   {
       resi=.0;
       visc=.0;
   }
   else if (icase!=1)
   {
       if (myid == 0) cout <<"Unknown icase "<<icase<<endl;
       MPI_Finalize();
       return 3;
   }
   if (myid == 0) args.PrintOptions(cout);

   if (use_petsc)
   {
      MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
   }

   //+++++Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //++++Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    backward Euler methods are available.
   PCSolver *ode_solver=NULL;
   ODESolver *ode_solver2=NULL;
   bool explicitSolve=false;
   switch (ode_solver_type)
   {
      //Explicit methods (first-order Predictor-Corrector)
      case 2: ode_solver = new PCSolver; explicitSolve = true; break;
      //Implict L-stable methods 
      case 1: ode_solver2 = new BackwardEulerSolver; break;
      case 3: ode_solver2 = new SDIRK23Solver(2); break;
      case 4: ode_solver2 = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 12: ode_solver2 = new ImplicitMidpointSolver; break;
      case 13: ode_solver2 = new SDIRK23Solver; break;
      case 14: ode_solver2 = new SDIRK34Solver; break;
     default:
         if (myid == 0) cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         MPI_Finalize();
         return 3;
   }

   //++++++Refine the mesh to increase the resolution.    
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   //+++++Define the vector finite element spaces representing  [Psi, Phi, w]
   // in block vector bv, with offsets given by the fe_offset array.
   // All my fespace is 1D but the problem is multi-dimensional
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll); 

   HYPRE_Int global_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of total scalar unknowns: " << global_size << endl;
   }

   int fe_size = fespace.TrueVSize();
   Array<int> fe_offset(4);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;

   BlockVector vx(fe_offset);
   ParGridFunction psi, phi, w, psiBack(&fespace), psiPer(&fespace);
   phi.MakeTRef(&fespace, vx, fe_offset[0]);
   psi.MakeTRef(&fespace, vx, fe_offset[1]);
     w.MakeTRef(&fespace, vx, fe_offset[2]);

   //+++++Set the initial conditions, and the boundary conditions
   FunctionCoefficient phiInit(InitialPhi);
   phi.ProjectCoefficient(phiInit);
   phi.SetTrueVector();

   if (icase==1)
   {
        FunctionCoefficient psiInit(InitialPsi);
        psi.ProjectCoefficient(psiInit);
   }
   else if (icase==2)
   {
        FunctionCoefficient psiInit2(InitialPsi2);
        psi.ProjectCoefficient(psiInit2);
   }
   else if (icase==3)
   {
        FunctionCoefficient psiInit3(InitialPsi3);
        psi.ProjectCoefficient(psiInit3);
   }
   psi.SetTrueVector();

   FunctionCoefficient wInit(InitialW);
   w.ProjectCoefficient(wInit);
   w.SetTrueVector();
   
   //this step is necessary to make sure unknows are updated!
   phi.SetFromTrueVector(); psi.SetFromTrueVector(); w.SetFromTrueVector();

   //Set the background psi
   if (icase==1)
   {
        FunctionCoefficient psi0(BackPsi);
        psiBack.ProjectCoefficient(psi0);
   }
   else if (icase==2)
   {
        FunctionCoefficient psi02(BackPsi2);
        psiBack.ProjectCoefficient(psi02);
   }
   else if (icase==3)
   {
        FunctionCoefficient psi03(BackPsi3);
        psiBack.ProjectCoefficient(psi03);
   }
   psiBack.SetTrueVector();

   //++++++this is a periodic boundary condition in x and Direchlet in y 
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;  //set attribute 1 to Direchlet boundary fixed
   if(ess_bdr.Size()!=1 || false)
   {
    if (myid==0) cout <<"ess_bdr size should be 1 but it is "<<ess_bdr.Size()<<endl;
    delete ode_solver;
    delete ode_solver2;
    delete mesh;
    delete pmesh;
    MPI_Finalize();
    return 2;
   }

   //++++Initialize the MHD operator, the GLVis visualization    
   ResistiveMHDOperator oper(fespace, ess_bdr, visc, resi, use_petsc, use_factory);
   if (icase==2)  //add the source term
   {
       FunctionCoefficient e0(E0rhs);
       oper.SetRHSEfield(e0);
   }
   else if (icase==3)     
   {
       FunctionCoefficient e0(E0rhs3);
       oper.SetRHSEfield(e0);
   }

   //set initial J
   if (icase==1)
   {
        FunctionCoefficient jInit(InitialJ);
        oper.SetInitialJ(jInit);
   }
   else if (icase==2)
   {
        FunctionCoefficient jInit2(InitialJ2);
        oper.SetInitialJ(jInit2);
   }
   else if (icase==3)
   {
        FunctionCoefficient jInit3(InitialJ3);
        oper.SetInitialJ(jInit3);
   }
   oper.BindingGF(vx);

   socketstream vis_phi;
   subtract(psi,psiBack,psiPer);
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_phi.open(vishost, visport);
      if (!vis_phi)
      {
          if (myid==0)
          {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
          }
         visualization = false;
      }
      else
      {
         vis_phi << "parallel " << num_procs << " " << myid << "\n";
         vis_phi.precision(8);
         vis_phi << "solution\n" << *pmesh << psiPer;
         vis_phi << "window_size 800 800\n"<< "window_title '" << "psi per'" << "keys cm\n";
         vis_phi << flush;

         MPI_Barrier(pmesh->GetComm());
      }
   }

   double t = 0.0;
   oper.SetTime(t);
   if (explicitSolve)
      ode_solver->Init(oper);
   else
      ode_solver2->Init(oper);

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (icase==1)
      {
        dc = new VisItDataCollection("case1", pmesh);
        dc->RegisterField("psiPer", &psiPer);
      }
      else if (icase==2)
      {
        dc = new VisItDataCollection("case2", pmesh);
        dc->RegisterField("psiPer", &psiPer);
        dc->RegisterField("psi", &psi);
        dc->RegisterField("phi", &phi);
        dc->RegisterField("omega", &w);
      }
      else
      {
        dc = new VisItDataCollection("case3", pmesh);
        dc->RegisterField("psiPer", &psiPer);
        dc->RegisterField("psi", &psi);
        dc->RegisterField("phi", &phi);
        dc->RegisterField("omega", &w);
      }

      bool par_format = false;
      dc->SetFormat(!par_format ?
                      DataCollection::SERIAL_FORMAT :
                      DataCollection::PARALLEL_FORMAT);
      dc->SetPrecision(8);
      dc->SetCycle(0);
      dc->SetTime(t);
      dc->Save();
   }

   MPI_Barrier(MPI_COMM_WORLD); 
   double start = MPI_Wtime();

   //++++Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      if (explicitSolve)
      {
         //---Predictor stage---
         //assemble the nonlinear terms
         phi.SetFromTrueVector(); oper.assembleNv(&phi);
         psi.SetFromTrueVector(); oper.assembleNb(&psi);
         ode_solver->StepP(vx, t, dt_real);

         //---Corrector stage---
         //assemble the nonlinear terms (only psi is updated)
         psi.SetFromTrueVector(); oper.assembleNb(&psi);
         ode_solver->Step(vx, t, dt_real);
         oper.UpdatePhi(vx);
      }
      else
      {
         ode_solver2->Step(vx, t, dt_real);
      }

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid==0) cout << "step " << ti << ", t = " << t <<endl;
         psi.SetFromTrueVector();
         subtract(psi,psiBack,psiPer);

         if (visualization)
         {
            if(icase!=3)
            {
                vis_phi << "parallel " << num_procs << " " << myid << "\n";
                vis_phi << "solution\n" << *pmesh << psiPer;
            }
            else
            {
                vis_phi << "parallel " << num_procs << " " << myid << "\n";
                vis_phi << "solution\n" << *pmesh << psi;
            }

            if (icase==1) 
            {
                vis_phi << "valuerange -.001 .001\n" << flush;
            }
            else
            {
                vis_phi << flush;
            }
         }

         if (visit)
         {
            if (icase!=1)
            {
              phi.SetFromTrueVector();
              w.SetFromTrueVector();
            }
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }

   }

   MPI_Barrier(MPI_COMM_WORLD); 
   double end = MPI_Wtime();

   //++++++Save the solutions.
   {
      phi.SetFromTrueVector(); psi.SetFromTrueVector(); w.SetFromTrueVector();

      ostringstream mesh_name, phi_name, psi_name, w_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      phi_name << "sol_phi." << setfill('0') << setw(6) << myid;
      psi_name << "sol_psi." << setfill('0') << setw(6) << myid;
      w_name << "sol_omega." << setfill('0') << setw(6) << myid;

      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(8);
      pmesh->Print(omesh);

      ofstream osol(phi_name.str().c_str());
      osol.precision(8);
      phi.Save(osol);

      ofstream osol3(psi_name.str().c_str());
      osol3.precision(8);
      psi.Save(osol3);

      ofstream osol4(w_name.str().c_str());
      osol4.precision(8);
      w.Save(osol4);
   }

   if (myid == 0) 
   { 
       cout <<"######Runtime = "<<end-start<<" ######"<<endl;
   }

   //+++++Free the used memory.
   delete ode_solver;
   delete ode_solver2;
   delete pmesh;
   delete dc;

   oper.DestroyHypre();

   if (use_petsc) { MFEMFinalizePetsc(); }

   MPI_Finalize();

   return 0;
}



