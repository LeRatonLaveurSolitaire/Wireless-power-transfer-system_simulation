import csv

def main() -> None :

    file_path = "exp_data\data_row.csv"
    output_file = "exp_data\data.csv"
    Vcc = 10
    wanted_freq = 1_000_000
    time,voltage,current,prbs = [],[],[],[]

    with open(file_path,newline='\n') as csv_file:
        
        csv_reader =csv.reader(csv_file,delimiter=',')

        for i,row in enumerate(csv_reader):   
            if i == 1:
                sampling_f = int(1/float(row[-1]))
            if i > 1 and not((i-2) % (sampling_f/wanted_freq)):
                time.append(int(row[0])/sampling_f)
                voltage.append((int(float(row[1]) > 1) - int(float(row[2]) > 1))*Vcc)
                current.append(float(row[3]))
                prbs.append(int(float(row[4]) > 2))


    with open(output_file,'w',newline='\n') as csv_file:
        csv_writer = csv.writer(csv_file,delimiter=',')
        for i in range(len(time)):
            csv_writer.writerow([time[i],voltage[i],current[i],prbs[i]])

if __name__ == "__main__":
    main()