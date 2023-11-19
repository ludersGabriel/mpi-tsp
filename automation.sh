#!/bin/bash

# Setting default directory
DIR="${1:-best}"

# File headers
echo "Size,Seq,2,4,8" > output/avg_times.csv
echo "Size,Seq,2,4,8" > output/stddev_times.csv
echo "Size,Seq,2,4,8" > output/parallel_times.csv
echo "Size,Seq,2,4,8" > output/pure_seq.csv

for size in {14..19}; do
    # Initializing an array for storing the results
    declare -A avg_results
    declare -A stddev_results
    declare -A parallel_avg_results
    declare -A pure_seq_results

    # Running for sequential and then for each core count
    for cores in 1 2 4 8; do
        declare -A program
        if [ $cores -eq 1 ]; then
           program="./tsp"
        else
            program="mpirun -np $cores mpi-tsp"
        fi

        echo "running '$program' $size $cores"

        # echo "$cores" > $DIR/temp.in
        # tail -n +2 $DIR/$size.in >> $DIR/temp.in

        cp $DIR/$size.in $DIR/temp.in

        # Get the initial time to determine the number of runs
        tmp=$($program < $DIR/temp.in 2>&1)

        initial_time=$(echo "$tmp" | grep -oP 'Total: \K\S+')
        initial_parallel_time=$(echo "$tmp" | grep -oP 'Parallel: \K\S+')
        initial_pure_seq_time=$(echo "$tmp" | grep -oP 'Pure_seq: \K\S+')

        all_times[1]=$initial_time

        # Decide on the number of runs based on the initial time
        if [ $(echo "$initial_time < 60" | bc) -eq 1 ]; then
            runs=20
        elif [ $(echo "$initial_time >= 60 && $initial_time < 300" | bc) -eq 1 ]; then
            runs=15
        elif [ $(echo "$initial_time >= 300 && $initial_time < 600" | bc) -eq 1 ]; then
            runs=10
        elif [ $(echo "$initial_time >= 600 && $initial_time < 1800" | bc) -eq 1 ]; then
            runs=5
        else
            runs=1
        fi

        # Calculate average and standard deviation for the remaining runs
        total_time=$initial_time
        total_parallel_time=$initial_parallel_time
        total_pure_seq_time=$initial_pure_seq_time
        for ((i=2; i<=$runs; i++)); do
            tmp=$($program < $DIR/temp.in 2>&1)
            real_run_time=$(echo $tmp | grep -oP 'Total: \K\S+')
            total_time=$(echo "$total_time + $real_run_time" | bc -l)
            all_times[$i]=$real_run_time

            cur_parallel_time=$(echo "$tmp" | grep -oP 'Parallel: \K\S+')
            total_parallel_time=$(echo "$total_parallel_time + $cur_parallel_time" | bc -l)

            cur_pure_seq_time=$(echo "$tmp" | grep -oP 'Pure_seq: \K\S+')
            total_pure_seq_time=$(echo "$total_pure_seq_time + $cur_pure_seq_time" | bc -l)
        done

        avg=$(echo "$total_time / $runs" | bc -l)
        avg_results[$cores]=$avg

        parallel_avg=$(echo "$total_parallel_time / $runs" | bc -l)
        parallel_avg_results[$cores]=$parallel_avg

        pure_seq_avg=$(echo "$total_pure_seq_time / $runs" | bc -l)
        pure_seq_results[$cores]=$pure_seq_avg

        # Calculate standard deviation for total times only
        sum_sq=0
        for t in "${all_times[@]}"; do
            diff=$(echo "$t - $avg" | bc -l)
            diff_sq=$(echo "$diff * $diff" | bc -l)
            sum_sq=$(echo "$sum_sq + $diff_sq" | bc -l)
        done

        stddev=$(bc -l <<< "scale=5; sqrt($sum_sq/$runs)")
        stddev_results[$cores]=$stddev

        unset all_times

        echo "Finished $size $cores"
        echo
    done

    # Write to CSV
    echo "$size,${avg_results[1]},${avg_results[2]},${avg_results[4]},${avg_results[8]}" >> output/avg_times.csv
    echo "$size,${stddev_results[1]},${stddev_results[2]},${stddev_results[4]},${stddev_results[8]}" >> output/stddev_times.csv
    echo "$size,${parallel_avg_results[1]},${parallel_avg_results[2]},${parallel_avg_results[4]},${parallel_avg_results[8]}" >> output/parallel_times.csv
    echo "$size,${pure_seq_results[1]},${pure_seq_results[2]},${pure_seq_results[4]},${pure_seq_results[8]}" >> output/pure_seq.csv

    unset avg_results
    unset stddev_results
    unset parallel_avg_results
    unset pure_seq_results
done
