#!/bin/bash

n_values=(5)
r_values=(1)
p_values=(1.0)
rtime_flags=(false)

output_file="results.txt"
> "$output_file"

for n in "${n_values[@]}"; do
  for r in "${r_values[@]}"; do
    for p in "${p_values[@]}"; do
      for rtime in "${rtime_flags[@]}"; do

        cmd="python3 src/federated_learning.py -n $n -r $r -p $p"
        if [ "$rtime" = true ]; then
          cmd="$cmd -rtime"
        fi

        echo "Running: $cmd"

        # catch last line
        last_line=$($cmd 2>&1 | tail -n 1)

        echo "$cmd -> $last_line" >> "$output_file"

      done
    done
  done
done

echo "Completed. Results in $output_file."
