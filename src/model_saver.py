import csv
import json
import pickle

class ModelSaver:
    @staticmethod

    #save_model will take the object (model) and a file path as arguments. Saving the instance of the object at the end of given file path
    def save_model(model, filepath, format):
        coefficients = model.coefficients
        if format == 'csv': #if desired format is csv:
            with open(filepath, 'w', newline='') as file:   #write a file to the designated filepath using csv.writer
                writer = csv.writer(file)
                writer.writerow(coefficients)
        elif format == 'json':  #if desired format is json, use json.dump to write a file to the designated filepath
            with open(filepath, 'w') as file:
                json.dump(coefficients.tolist(), file)
        elif format == 'pickle': #if desired format is pickle, use pickle.dump to  write file to designated filepath
            with open(filepath, 'wb') as file:
                pickle.dump(coefficients, file)
        else:
            raise ValueError("Unsupported format. Please use 'csv', 'json', or 'pickle'.")  #any other format will throw an exception

    @staticmethod
    #load_model loads the instance of the model from a place in the files. Directed by the given file path
    def load_model(filepath, format):
        if format == 'csv': #check file, if csv format, use csv.read to load
            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                coefficients = next(reader)
                return [float(coef) for coef in coefficients]
        elif format == 'json': #check file, if json format, use json.load to load
            with open(filepath, 'r') as file:
                return json.load(file)
        elif format == 'pickle':    #check file, if pickle format, use pickle.load to load
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        else:
            raise ValueError("Unsupported format. Please use 'csv', 'json', or 'pickle'.")
