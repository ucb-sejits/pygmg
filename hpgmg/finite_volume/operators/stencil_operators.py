/* A simple test harness for memory alloction. */

#include "mm_alloc.h"
#include <stdio.h>



int main(int argc, char **argv)
{
	//sanity check
    int *data;
    data = (int*) mm_malloc(4);
    data[0] = 1;
    mm_free(data);
    printf("malloc sanity test successful!\n");

    //test linked list
    struct s_block *root;   
    root = (struct s_block *) malloc( sizeof(struct s_block) ); 
    root->next =  0;
    root->free = 5;
    printf("linked list test should say 5: %i\n",root->free );
    free(root);

    //test pad funciton
    printf("should say 44+4: %i \n", pad(4));
    printf("should say 48+4: %i \n", pad(7));


    //test basic allocation
    int size = 15; //56
    flist free_list = {0};

    if (free_list.head ==  NULL) {
		free_list.head = sbrk(pad(size)); 
		free_list.head->size = size; 
		free_list.head->free=1; 
	}
	printf("should say %i: %i\n",size, free_list.head->size );
	printf("should say 0: %i \n", free_list.head==FREE);

	size = 





	free(free_list);

    
    return 0;
}
