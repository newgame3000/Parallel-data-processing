#include <iostream>
#include <time.h>
#include "mpi.h"
#include <string>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace std;

//k * (nx * ny) + j * nx + i

#define _i(i, j, k) (((k) + 1) * (bx + 2) * (by + 2) + ((j) + 1) * (bx + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * sx * sy + (j) * sx + (i))

int main(int argc, char *argv[]) {
    char proc_name[MPI_MAX_PROCESSOR_NAME];

    int id, numproc, proc_name_len; 

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Get_processor_name(proc_name, &proc_name_len);  

    int sx, sy, sz;
    int bx, by, bz;
    string out_name;
    double eps;
    double lx, ly, lz;
    double u_up, u_down, u_front, u_back, u_left, u_right, u_0;
    int ib, jb, kb; 

    ofstream of;

    if (id == 0) {
        
        cin >> sx >> sy >> sz;
        cin >> bx >> by >> bz;
        cin >> out_name;
        cin >> eps;
        cin >> lx >> ly >> lz;
        cin >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back >> u_0;


        of.open(out_name);
        of << scientific << setprecision(6);
    }

    //down up - z
    //front back - y
    //left right - x
    MPI_Barrier(MPI_COMM_WORLD);
  
    MPI_Bcast(&sx, 1, MPI_INT, 0, MPI_COMM_WORLD);    
    MPI_Bcast(&sy, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    MPI_Bcast(&bx, 1, MPI_INT, 0, MPI_COMM_WORLD);    
    MPI_Bcast(&by, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&bz, 1, MPI_INT, 0, MPI_COMM_WORLD);   

    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    double *data, *temp, *next;

    data = (double *)malloc(sizeof(double) * (bx + 2) * (by + 2) * (bz + 2));    
    next = (double *)malloc(sizeof(double) * (bx + 2) * (by + 2) * (bz + 2));

    double hx = lx / (sx * bx); 
    double hy = ly / (sy * by);
    double hz = lz / (sz * bz);

    double *buffx, *buffy, *buffz;

    buffx = (double *)malloc(sizeof(double) * by * bz);
    buffy = (double *)malloc(sizeof(double) * bx * bz);
    buffz = (double *)malloc(sizeof(double) * bx * by);



    kb = id / (sx * sy);
    jb = id % (sx * sy) / sx;
    ib = id % (sx * sy) % sx;


    for(int i = 0; i < bx; ++i) { 
        for(int j = 0; j < by; ++j) {
            for(int k = 0; k < bz; ++k) {
                data[_i(i, j, k)] = u_0;
            }
        }
    }

    bool work = true;
    
    //down up - z
    //front back - y
    //left right - x


    while(work) {

        MPI_Barrier(MPI_COMM_WORLD);

        //вверх вправо назад отправляем, снизу слева спереди получаем

        //Вправо отправляем, то есть по оси x

        if  (ib + 1 < sx) {
            for (int i = 0; i < by; ++i) {
                for (int j = 0; j < bz; ++j) {
                    buffx[j * by + i] = data[_i(bx - 1, i, j)];
                }
            }

            MPI_Send(buffx, by * bz, MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
        }

        //Назад отправляем, то есть по оси y
        if (jb + 1 < sy) { 
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < bz; ++j) {
                    buffy[j * bx + i] = data[_i(i, by - 1, j)];
                }
            }

            MPI_Send(buffy, bx * bz, MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
        }

        //Вверх отправляем, то есть по оси z
        if (kb + 1 < sz) { 
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < by; ++j) {
                    buffz[j * bx + i] = data[_i(i, j, bz - 1)];
                }
            }

            MPI_Send(buffz, bx * by, MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
        }


        //Получаем
        //Получаем слева, то есть по оси x
        if (ib > 0) {
            MPI_Recv(buffx, by * bz, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);

            for (int i = 0; i < by; ++i) {
                for (int j = 0; j < bz; ++j) {
                    data[_i(-1, i, j)] = buffx[j * by + i];
                }
            }
        } else {
            for (int i = 0; i < by; ++i) {
                for (int j = 0; j < bz; ++j) {
                    data[_i(-1, i, j)] = u_left;
                }
            }
        }

         

        //Получаем спереди, ось y
        if (jb > 0) {

            MPI_Recv(buffy, bx * bz, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);

            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < bz; ++j) {
                    data[_i(i, -1, j)] = buffy[j * bx + i];
                }
            }
        } else {
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < bz; ++j) {
                    data[_i(i, -1, j)] = u_front;
                }
            }
        }

        //Получаем снизу, ось z
        if (kb > 0) {

            MPI_Recv(buffz, bx * by, MPI_DOUBLE, _ib(ib, jb, kb - 1),  _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);

            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < by; ++j) {
                    data[_i(i, j, -1)] = buffz[j * bx + i];
                }
            }
        } else {
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < by; ++j) {
                    data[_i(i, j, -1)] = u_down;
                }
            }
        }


        MPI_Barrier(MPI_COMM_WORLD);
        //вниз влево вперёд отправляем, сверху справа сзади получаем

        //отправляем влево, то есть по оси x
        if (ib > 0) {
            for (int i = 0; i < by; ++i) {
                for (int j = 0; j < bz; ++j) {
                    buffx[j * by + i] = data[_i(0, i, j)];
                }
            }

            MPI_Send(buffx, by * bz, MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
        }

        //отправляем вперед, ось y
        if (jb > 0) {
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < bz; ++j) {
                    buffy[j * bx + i] = data[_i(i, 0, j)];
                }
            }

            MPI_Send(buffy, bx * bz, MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
        }

        //Отправляем вниз, ось z
        if (kb > 0) {
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < by; ++j) {
                    buffz[j * bx + i] = data[_i(i, j, 0)];
        
                }
            }
            MPI_Send(buffz, bx * by, MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
        }

        //Получаем 
        // Получаем справа, ось х
        if (ib + 1 < sx) {
            MPI_Recv(buffx, by * bz, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
        
            for (int i = 0; i < by; ++i) {
                for (int j = 0; j < bz; ++j) {
                    data[_i(bx, i, j)] = buffx[j * by + i] ;
                }
            }

        } else {
            for (int i = 0; i < by; ++i) {
                for (int j = 0; j < bz; ++j) {
                    data[_i(bx, i, j)] = u_right;
                }
            }
        }

        //Получаем сзади, ось y
         if (jb + 1 < sy) { 
            MPI_Recv(buffy, bx * bz, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
        
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < bz; ++j) {
                    data[_i(i, by, j)] = buffy[j * bx + i];
                }
            }

        } else {
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < bz; ++j) {
                    data[_i(i, by, j)] = u_back;
                }
            }
        }

        //Получаем сверху, ось z
        if (kb + 1 < sz) { 
            MPI_Recv(buffz, bx * by, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
        
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < by; ++j) {
                   data[_i(i, j, bz)] = buffz[j * bx + i];
                }
            }
        } else {
            for (int i = 0; i < bx; ++i) {
                for (int j = 0; j < by; ++j) {
                   data[_i(i, j, bz)] = u_up;
                }
            }
        }


        //Дело сделано
        MPI_Barrier(MPI_COMM_WORLD);

        double max = -1;
        double r = 0;

        for (int i = 0; i < bx; ++i) {
            for (int j = 0; j < by; ++j) {
                for (int k = 0; k < bz; ++k) {
                    next[_i(i, j, k)] = ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx)  + 
                                        (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy)   +
                                        (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz))  / 
                                        (2 * (1 / (hx * hx) + 1 / (hy * hy) + 1 / (hz * hz)));

                    r = abs(next[_i(i, j, k)] - data[_i(i, j, k)]);
                    if (r > max) {
                        max = r;
                    }

                }
            }
        }

        work = false;

        MPI_Barrier(MPI_COMM_WORLD);    

        MPI_Allreduce(&max, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (r > eps) {
           work = true;
        }

        temp = next;
        next = data;
        data = temp;
    }


    double *resbuff;
    resbuff = (double *)malloc(sizeof(double) * bx);

    if (id != 0) {  
        for (int k = 0; k < bz; ++k) {
            for (int j = 0; j < by; ++j) {
                for (int i = 0; i < bx; ++i) {

                    resbuff[i] = data[_i(i, j, k)];
        
                }

                MPI_Send(resbuff, bx, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
        }

    } else {
        for (int k = 0; k < sz; ++k) {
            for (int kk = 0; kk < bz; ++kk) {
                for (int j = 0; j < sy; ++j) {
                    for (int jj = 0; jj < by; ++jj) {
                        for (int i = 0; i < sx; ++i) {
                            if (_ib(i, j, k) != 0) {
                                MPI_Recv(resbuff, bx, MPI_DOUBLE, _ib(i, j, k), _ib(i, j, k), MPI_COMM_WORLD, &status);
                                for (int p = 0; p < bx; ++p) {
                                    of << resbuff[p] << " ";
                                }
                            } else {
                                for (int p = 0; p < bx; ++p) {
                                    of << data[_i(p, jj, kk)] << " ";
                                }
                            }
                        }
                        of << endl;
                    }
                    
                }
                of << endl;
            }
        }
    }


    free(data);
    free(next);
    free(buffx);
    free(buffy);
    free(buffz);
    MPI_Finalize();
    return 0;
}