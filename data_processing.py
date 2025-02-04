import os
import pandas as pd
import json
import shutil
from functools import reduce
import operator

def recursive_folder_look(folder, str_filter=""):
    for e in os.listdir(folder):
        file = os.path.join(folder, e)
        if os.path.isdir(file):
            yield from recursive_folder_look(file, str_filter)
        elif str_filter in e:
            yield file
    

def get_load_files(folder, str_filter) -> list[pd.DataFrame]:
    files = []
    for element in recursive_folder_look(folder, str_filter):
        print("Processing Element", element)
        files.append(pd.read_csv(element))

    return files

def run_training_data_analysis_ms(folder, sample_rate="500ms"):


    # Convert 'created_at' to datetime for the second dataset
    transfer_log_data['created_at'] = pd.to_datetime(transfer_log_data['created_at'])

    # Set 'created_at' as the index for resampling
    transfer_log_data.set_index('created_at', inplace=True)

    # Resample data into 500ms intervals
    # Aggregating with sum for `data_transfer_len` and count for `id` to show activity
    aggregated_transfer_data = transfer_log_data.resample('500ms').agg({
        'data_transfer_len': 'sum',     # Total data transferred in the interval
        'id': 'count',                 # Count of log entries in the interval
        'direction_to_server': 'mean'  # Proportion of entries directed to the server
    })

    # Reset index for better readability
    aggregated_transfer_data.reset_index(inplace=True)


def run_training_data_analysis_count_rows(folder, chunk_size):

    grouped = []

    for transfer_log_data in get_load_files(folder, "DataTransferLog.csv"):

        grouped_transfer_data = transfer_log_data.groupby(
        transfer_log_data.reset_index().index // chunk_size
        ).agg({
            'data_transfer_len': 'sum',
            'id': 'count'
        }).reset_index(drop=True)
        grouped.append(grouped_transfer_data)

    concat = pd.concat(grouped)
    result = concat.groupby(concat.index).agg({
            'data_transfer_len': 'sum', 
            'id': 'sum', 
    })

    result.to_csv(os.path.join(folder, "agg_Data_Transfer_Log.csv"))
    return result


def run_client_average(folder, mode_force):
    files = get_load_files(folder,"TrainingLog.csv")
    results = []

    for df in files:
        # if mode_force:
        #     df = df[df['mode'] == mode_force]
        # else:
        #     df = df[df['mode'] != "reset"]

        df.info()
        df['Group'] = df["client_id"] % 2

        # df.set_index('created_at', inplace=True)
        # sampled = df.resample(sample_rate)

        aggregated_data = df.groupby(["Group"]).agg({
            'loss': 'mean', 
            'epoch': 'max', 
            'server_epoch': 'max',  
            'training_time': 'mean',    
            'last_communication_time': 'mean',  
            'last_whole_training_time': 'mean',
            'id': 'count'                 
        })
        print(aggregated_data)
        aggregated_data.reset_index(inplace=True)
        results.append(aggregated_data)

    concat = pd.concat(results)
    result = concat.groupby(concat.index).agg({
            'loss': 'mean', 
            'epoch': 'max', 
            'server_epoch': 'max',  
            'training_time': 'mean',    
            'last_communication_time': 'mean',  
            'last_whole_training_time': 'mean',
            'id': 'sum'
    })

    result.to_csv(os.path.join(folder, "agg_Training_Log_clients.csv"))
    return result

    

def run_prediction_log_analysis(folder, chunkspred):

    dfs = get_load_files(folder, "PredictionLog.csv")
    finals = []
    for df in dfs: 
        df['epoch_interval'] = df['client_epoch'] // chunkspred
        
        print(df)
        result = (

            df.assign(correct_prediction=(df['prediction_result'] == df['expected_result']))
            .groupby(['epoch_interval'])
            .agg(correct_prediction = ('correct_prediction', 'sum'), counter_predict= ('id', 'count'))
            .reset_index()
        )
        
        result.info()
        finals.append(result)


    concat = pd.concat(finals)

    concat.groupby("epoch_interval").agg({
        'correct_prediction': 'mean',
        'counter_predict': 'min'
    })
    print('counter', result)

    result.to_csv(os.path.join(folder, "agg_Prediction_Log.csv"))
    return result


def get_avrg_len(folder):
    files = get_load_files(folder,"TrainingLog.csv")
    results = []
    for df in files:
        results.append(len(df))
    
    return sum(results)/len(results)

def run_avrg_loss_calculation_on_testing(folder, sample_rate=5, mode_force=None):

    files = get_load_files(folder,"TrainingLog.csv")
    
    results = []
    for df in files:
        # df['created_at'] = pd.to_datetime(df['created_at'], format="mixed")
        if mode_force:
            df = df[df['mode'] == mode_force]
        else:
            df = df[df['mode'] != "reset"]

        df.info()
        df['Group'] = df["id"] // sample_rate

        # df.set_index('created_at', inplace=True)
        # sampled = df.resample(sample_rate)

        aggregated_data = df.groupby(["Group"]).agg({
            'loss': 'mean', 
            'epoch': 'max', 
            'server_epoch': 'max',  
            'training_time': 'mean',    
            'last_communication_time': 'mean',  
            'last_whole_training_time': 'mean',
            'id': 'count'                 
        })
        print(aggregated_data)
        aggregated_data.reset_index(inplace=True)
        results.append(aggregated_data)

    concat = pd.concat(results)
    result = concat.groupby(concat.index).agg({
            'loss': 'mean', 
            'epoch': 'max', 
            'server_epoch': 'max',  
            'training_time': 'mean',    
            'last_communication_time': 'mean',  
            'last_whole_training_time': 'mean',
            'id': 'sum'
    })

    result.to_csv(os.path.join(folder, "agg_Training_Log.csv"))

    return result

def run_details_aggregate(folder):
    
    files = []
    times = []
    number_of_elements = 0
    for element in recursive_folder_look(folder, "details.json"):
        number_of_elements += 1
        print("Processing Element", element)
        with open(element, encoding="utf-8") as f:
            details = json.load(f)
        
        times.append(details['logs_timer'])
        
    d = {"mean_time": sum(times)/len(times), "number_of_tests": number_of_elements}
    with open(os.path.join(folder, "agg_details_json.json"), "w") as f:
        json.dump(d, f)

    return d
        
    

differentiate_fields = ["selected_model", "clients", "disable_client_side_drift_detection", "disable_server_side_drift_detection", "drift_type", "check_mode", "check_testing_repeating", "prediction_errors_count", 'start_deviation_target', 'server_zscore_deviation', 'server_error_threshold', 'server_filter_last_tests']
sanity_values_check = {
    "predict_epochs_swap": 100,
    "max_predict_epoch": 1000,
    'target_loss': 0.4,
    "mode": "normal_runner",
}

def run_grouping_algo(folder, target_folder):
    if os.path.exists(target_folder) and input("Do you want to delete target? y/n").lower() == "y":
        shutil.rmtree(target_folder)
        

    l = []
    for f in os.listdir(folder):
        base_path = os.path.abspath(os.path.join(folder, f))        
        print(base_path)

        if not os.path.isdir(base_path):
            print("not a dir")
            continue

        def j(*paths):
            return os.path.join(base_path, *paths)
        
        with open(j("server_data/details.json"), encoding="utf-8") as loaded_f:
            details = json.load(loaded_f)

        failed = False
        for key, value in sanity_values_check.items():
            if details['all_runner_options'][key] != value:
                print(f"{key} Values {details['all_runner_options'][key]} don't match {value}! Skipping...")
                failed = True
                break

        if failed:
            continue

        local_list = [base_path, f, details['all_runner_options']['description']]
        for field in differentiate_fields:
            if field not in details['all_runner_options']:
                print(f"No field {field}! That means test is old! skippping")
                failed = True
                break
                
            val = details['all_runner_options'][field]
            if not isinstance(val, (int, str, float, bool)):
                val = str(val)

            local_list.append(val)
        
        if failed:
            continue
        l.append(local_list)

    print(l)
    df = pd.DataFrame(l, columns=["folder_abs", 'folder_name', "description"] + differentiate_fields)
    df.to_csv("record_of_agg.csv")
    df.info()

    grouping = df.groupby(differentiate_fields)

    target = target_folder
    os.makedirs(target, exist_ok=True)

    for group_name, group_data in grouping:
        folder_name = (";".join([str(x) for x in group_name]))
        copy_target = os.path.join(target, folder_name)
        os.makedirs(copy_target, exist_ok=True)

        for index, elements in group_data.iterrows():
            print(elements['folder_name'])
            shutil.copytree(elements['folder_abs'], os.path.join(copy_target, elements['folder_name']))
            print(elements['folder_abs'])

    return df
    

def run_target_aggregator(agg_record, target_folder, interval, chunks, chunkspread, modeforce):
    target_list = []
    labels = None
    start_num = 0
    if isinstance(agg_record, str):
        df = pd.read_csv(agg_record)
        start_num = 1
    else:
        df = agg_record
    
    for folder in os.listdir(target_folder):
        f = os.path.join(target_folder, folder)
        
        avrg = run_avrg_loss_calculation_on_testing(f, sample_rate=interval, mode_force=modeforce)
        train = run_training_data_analysis_count_rows(f, chunk_size=chunks)
        pred = run_prediction_log_analysis(f, chunkspred=chunkspread)
        clients = run_client_average(f, mode_force=modeforce)

        details_data = run_details_aggregate(f)
        
        train_agg = train.agg({
            "data_transfer_len": "sum"
        })

        
        losses = avrg.agg(
            loss_max = ("loss", "max"),
            loss_min = ("loss", "min"),
            loss_avrg = ("loss", "mean"),
        )

        counter_of_all = avrg.agg(
            number_of_operations = ("id", "sum")

        )

        
        # print(pred)]
        print(pred)
        pred_calc = pred.agg(
            mean_success = ("correct_prediction", "mean"),
            worst_success = ("correct_prediction", "min"),
            mean_success_per_client = ("correct_prediction", "mean"),
        )

        counter_pred_calc = pred.agg(
            sum_of_all = ("counter_predict", "sum")
        )

        flatten_losses = reduce(operator.ior, [{**losses[key]} for key in losses.to_dict()], {})
        flatten_pred = reduce(operator.ior, [{**pred_calc[key]} for key in pred_calc.to_dict()], {})
        # flatten_counter_pred = reduce(operator.ior, [{**counter_pred_calc[key]} for key in counter_pred_calc.to_dict()], {})
        flatten_counter_pred = {"sum_of_all": get_avrg_len(f)}
        flatten_counter_of_all = reduce(operator.ior, [{**counter_of_all[key]} for key in counter_of_all.to_dict()], {})
        splitter = folder.split(";")
        d = {}
        for i in range(len(splitter)):
            d[df.columns[i+start_num+3]] = splitter[i]
        print(d)
        flatten_pred['mean_success_per_client'] = flatten_pred['mean_success_per_client']/int(d['clients'])

    
        print("Aggregated Data")
        
        clients_data = {
            "divisable_by_0_drifting": clients[clients.index == 0].iloc[0]['loss'],
            "not_divisable_by_0_not_drifting": clients[clients.index == 1].iloc[0]['loss'],
        }

        # print(flatten_losses)


        # print(d)

        l = {**d, **train_agg.to_dict(), **flatten_losses, **flatten_pred, **details_data, **flatten_counter_pred, **flatten_counter_of_all, **clients_data}
        if labels is None:
            labels = l.keys()
        target_list.append(l)

    df2 = pd.DataFrame(target_list, columns=labels)
    df2.to_csv("data_processing_completed.csv")
        

        # sum transferred data	Max Loss	Min Loss	Mean Loss	Avrg Loss	Avrg failed predictions



    




import argparse
if __name__ == "__main__":
    print("Processing files")


    parser = argparse.ArgumentParser("File Processing")
    parser.add_argument("folder", help="folder to process", type=str)
    parser.add_argument("--modeforce", help="filter only training data with given type", type=str, default=None)
    parser.add_argument("--interval", help="Division interval in str", type=int, default=5)
    parser.add_argument("--chunks", help="Chunks interval in int", type=int, default=5)
    parser.add_argument("--chunkspred", help="Chunks prediction interval in int", type=int, default=10)
    parser.add_argument("--mass", help="Run target data grouping", action="store_true")
    parser.add_argument("--target", help="target folder to process", type=str, default="target_folder")

    args = parser.parse_args()
    print(args.mass)

    if args.mass:
        agg_data = run_grouping_algo(args.folder, args.target)
        # agg_data = "record_of_agg.csv"
        run_target_aggregator(agg_data, args.target, args.interval, args.chunks, args.chunkspred, args.modeforce)
    
    else:
        run_avrg_loss_calculation_on_testing(args.folder, args.interval, args.modeforce)
        run_training_data_analysis_count_rows(args.folder, args.chunks)
        run_prediction_log_analysis(args.folder, args.chunkspred)

    