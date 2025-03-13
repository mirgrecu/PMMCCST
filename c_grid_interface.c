#include "closest/src/closest.h"
void find_nearest_(double *gmi_loc_data, double *dpr_loc_data, int *ngmi, int *ndpr, int *i_loc, double *d_loc)
{

int k = 4;
int n = *ndpr;


/* initialize the search */
cell_t *cell = cell_init( 2, *ngmi, gmi_loc_data, -1 );

int i_cell[k];
double d_cell[k];
/* cell_knearest returns
     in i_cell, the index for the points in the `data` array of the k-nearest neighbors to x
     in d_cell, the distances for those same points to x */
for (int i = 0; i < *ndpr; i++) {
    double x[2] = {dpr_loc_data[2*i], dpr_loc_data[i*2+1]};
    cell_knearest( cell, 1, x, k, &i_loc[4*i], &d_loc[4*i] );
}
cell_free(cell);
}