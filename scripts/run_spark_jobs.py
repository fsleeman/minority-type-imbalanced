from os import makedirs
from os.path import isdir, join


def main():
    input_data = '/home/ford/data/poker_test/poker10k.data'
    label_column = 'label'
    use_header = 'yes'
    resampling_methods = ['none', 'undersample', 'oversample', 'smote', 'smotePlus']
    output_directory = '/home/ford/data/poker_test/results'

    rw_path = '/home/ford/data/poker_test/minorityDf10k'

    jar_path = '/home/ford/repos/imbalanced-spark/target/scala-2.11/imbalanced-spark-assembly-0.1.jar'

    input_name = input_data.split('/')[-1]

    input_name = input_name[0:input_name.find('.')]

    k_values = [2, 5, 10]

    commands = []

    for index, k in enumerate(k_values):
        output_description = input_name
        if index == 0:
            rw = 'write'
        else:
            rw = 'read'

        current_directory = join(output_directory, output_description, 'k' + str(k))

        if isdir(current_directory):
            print('exists')
        else:
            print('does not exists')
            makedirs(current_directory)

        print("***")
        print(current_directory)
        print(output_description)
        print("***")

        command = 'spark-submit --class edu.vcu.sleeman.Classifier --conf spark.master=local[*] ' \
                  + jar_path + ' ' + input_data + ' ' \
            + label_column  + ' ' + 'sparkNN yes ' + ' ' + output_directory + ' ' + output_description + ' ' + rw \
            + ' ' + rw_path + ' ' + str(k)

        for method in resampling_methods:
            command += ' ' + method

        print(command)
        commands.append(command)
    with open('commands.sh', 'w') as out_file:
        for command in commands:
            out_file.write(command + '\n')

    # /home/ford/data/poker_test/poker10k.data label sparkNN yes /home/ford/data/poker_test/results/ poker10k write /home/ford/data/poker/minorityDf 10

if __name__ == "__main__":
    main()