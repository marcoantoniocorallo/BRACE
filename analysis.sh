#!/bin/bash

n_values=(4 5 6)
r_values=(1)
p_values=(1 0.8 0.7)
b_values=(1 2 3)
rtime_flags=(true)

output_file="results.txt"
>> "$output_file"

echo "--------------------------------------------------------" >> "$output_file"

mkdir -p logs  # crea la directory per i log se non esiste

for n in "${n_values[@]}"; do
  for r in "${r_values[@]}"; do
    for p in "${p_values[@]}"; do
      for b in "${b_values[@]}"; do
        for rtime in "${rtime_flags[@]}"; do

          cmd="python3 src/federated_learning.py -n $n -r $r -p $p -b $b"
          if [ "$rtime" = true ]; then
            cmd="$cmd -rtime"
          fi

          echo "Running: $cmd"

          logfile="logs/n${n}_r${r}_p${p}_rtime${rtime}.log"
          $cmd > "$logfile" 2>&1

          last_line=$(tail -n 1 "$logfile")
          echo "$cmd -> $last_line" >> "$output_file"

          echo "Finished: $cmd"

        done
      done
    done
  done
done

echo "Completed. Results in $output_file."
