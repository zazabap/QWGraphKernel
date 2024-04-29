
from main import * 
import multiprocessing

def process_density_matrix(H):
    # Assuming resizeMatrix and getDensityMatrix are defined somewhere
    return getDensityMatrix(H, 1, 10, n_wires)

def process_in_parallel(H_list, n_wires):
    num_processes = multiprocessing.cpu_count()  # Get the number of CPU cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process each element of H_list in parallel
        rho_list = pool.map(process_density_matrix, H_list)
    return rho_list

if __name__ == "__main__":
    print("Time Evoluton")
    start_time = time.time()
    rho_list = process_in_parallel(H_list,n_wires)

    # for i in range(len(H_list)):
    #     rho_list.append(getDensityMatrix(H_list[i],1,10,n_wires),n_wires)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")