#!/bin/bash

make_clean=1   # 1 if cleaning is required
make___new=1   # 1 if programs need to be run

if (($make_clean==1)) ; then
    files_to_delete=("imag_prop" "real_prop" "tsurff" "tsurff_mpi" "isurfv" )

    for i in ${!files_to_delete[*]} ; do
        file=${files_to_delete[$i]}
        if [ -f $file ] ; then
#             printf "Deleting %s\n" "${files_to_delete[$i]}"
            rm $file
        fi
    done
    
#     printf "Deleting .dat files\n"
    rm -f ./wf/*.dat
	rm -f ./dat/*.dat
#     printf "Deleting .raw files\n"
    rm -f ./dat/*.raw
#     printf "Deleting .log files\n"
    rm -f ./dat/*.log
fi


if (($make___new==1)) ; then

    printf "Performing \"make clean\"\n"
    make clean

    printf "Making imag_prop\n"
    make imag_prop --silent
    ./imag_prop
    
    printf "Making real_prop\n"
    make real_prop --silent
    ./real_prop
fi
printf "Hasta la vista...\n"
exit 0