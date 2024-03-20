from train import train_and_save_model
from test import test_model

# Main function
# @argv: command line arguments
#    @argv[1]: 'train' or 'test'
#    @argv[2]: name of the model to be saved or tested
#    @argv[3]: 'HOG' or 'LBP' extractor types
def main(command_line_args):
    if len(command_line_args) < 2:
        print('Usage: python main.py train <name_of_the_model> <HOG|LBP> or python main.py test <name_of_the_model>')
        exit(1)

    elif command_line_args[1] == 'train':
        if len(command_line_args) < 4 or command_line_args[3].upper() not in ['HOG', 'LBP']:
            print('Usage: python main.py train <name_of_the_model> <HOG|LBP>')
            exit(1)
        train_and_save_model(extractor_type=command_line_args[3].upper(), model_name=command_line_args[2])

    elif command_line_args[1] == 'test':
        if len(command_line_args) < 3:
            print('Usage: python main.py test <name_of_the_model>')
            exit(1)
        test_model(model_name=command_line_args[2])

if __name__ == '__main__':
    import sys
    command_line_args = sys.argv
    main(command_line_args)
