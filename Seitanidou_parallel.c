#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

double Ixy(double x,double y,int Lx,int Ly){
  return x*(Lx-x)*y*(Ly-y);
}

double Vxy(double x,double y,int Lx,int Ly){
  return 1/2*x*(Lx-x)*y*(Ly-y);
}

double fxyt(double x,double y,double t,int Lx,int Ly){
  return 2*(1+1/2*t)*(y*(Ly-y)+x*(Lx-x));
}

double u_anal(double x,double y, double t, int Lx, int Ly){
  return x*(Lx-x)*y*(Ly-y)*(1+1/2*t);
}

int main(int argc, char const *argv[]) {
  int i,j,k,n,nThreads;
  int Nt=5;
  int T=20;
  int Nx, Ny=Nx=40;
  int Lx, Ly=Lx=10;
  double Cx,Cy,u,diff;
  double **un, **un1, *x, *y, *t, dx,dy,dt;
  double t1,t2;

  t1 = omp_get_wtime();

  FILE *fp1;
  fp1 = fopen("sol_num.dat","w");
  FILE *fp2;
  fp2 = fopen("sol_anal.dat","w");

  // Memory Allocation

  // un ->  i,j,n
  un = (double**) malloc(Nx*sizeof(double*));
  for (int i = 0; i < Nx; i++)
      un[i] = (double*) malloc(Ny*sizeof(double));

  // un1 -> i,j,n+1
  un1 = (double**) malloc(Nx*sizeof(double*));
  for (int i = 0; i < Nx; i++)
      un1[i] = (double*) malloc(Ny*sizeof(double));


  x = malloc(Nx*sizeof(double));
  y = malloc(Ny*sizeof(double));
  t = malloc(Nt*sizeof(double));

  // x,y mesh
  dx = (Lx-0.0)/(Nx-1);
  for(i=0;i<Nx;i++) x[i] = 0 + i*dx;

  dy = (Ly-0.0)/(Ny-1);
  for(j=0;j<Ny;j++) y[j] = 0 + j*dy;

  dt = (T-0.0)/(Nt-1);
  for(k=0;k<Nt;k++) t[k] = 0 + k*dt;

  // constants
  Cx = dt/dx;
  Cy = dt/dy;

  #pragma omp parallel private(i,j,n) shared(un,un1,u,diff,x,y,t,dx,dy,dt,Cx,Cy,Nx,Ny,Nt,Lx,Ly,fp1,fp2,nThreads) default(none)
  {
    nThreads = omp_get_num_threads();
    // initial values
    #pragma omp for collapse(2)
    for(i=0;i<Nx;i++)
      for(j=0;j<Ny;j++)
          un[i][j] = Ixy(x[i],y[j],Lx,Ly);

    // boundary conditions
    #pragma omp for
    for(i=0;i<Nx;i++) {
      un[i][0] = 0.0;
      un[i][Nx-1] = 0.0;
      un1[i][0] = 0.0;
      un1[i][Nx-1] = 0.0;
    }

    #pragma omp for
    for(j=0;j<Ny;j++) {
      un[0][j] = 0.0;
      un[Nx-1][j] = 0.0;
      un1[0][j] = 0.0;
      un1[Nx-1][j] = 0.0;
    }

    #pragma omp for collapse(3) schedule(static)
    for(n=0;n<Nt;n++){
      for(i=1;i<Nx-1;i++){
        for(j=1;j<Ny-1;j++){
          un1[i][j] = dt*Vxy(x[i],y[j],Lx,Ly)+un[i][j]+1/2*Cx*Cx*(un[i+1][j]-2*un[i][j]+un[i-1][j])
                    +1/2*Cy*Cy*(un[i][j+1]-2*un[i][j]+un[i][j-1]) + 1/2*dt*dt*fxyt(x[i],y[j],t[n],Lx,Ly);
          #pragma omp critical
          {
            u = u_anal(x[i],y[j],t[n],Lx,Ly);
            diff = fabs(u - un1[i][j]);
            //printf("%e\n",diff);

            //run for t[n]=0,5,10,20
            if(t[n]==0){
              fprintf(fp1,"%f %f %f\n",x[i],y[j],un1[i][j]);
              fprintf(fp2,"%f %f %f\n",x[i],y[j],u);
            }
          }
        }
      }
    }
  }

  printf("u[Nx/2][Ny/2] = %f\n", un1[Nx/2][Ny/2]);

  fclose(fp1);
  fclose(fp2);

  system("gnuplot -p plot_num.gp");
  system("gnuplot -p plot_anal.gp");

  for (i = 0; i < Nx; i++){
        free(un[i]);
      }
  free(un);

  for (i = 0; i < Nx; i++){
        free(un1[i]);
      }
  free(un1);

  free(x);
  free(y);
  free(t);

  t2 = omp_get_wtime();

  printf("For %d threads, time: %f\n",nThreads,t2-t1);

  return 0;
}
