import os
import pickle
import csv
import pandas as pd


logTextFile = "log.txt"



class LogClass:
        
    def __init__(self, logDir: str):
        """

        Args:
            logDir (str): The directory where this Log will be worked on
        
        """
        
        try:
            os.listdir(logDir)
            print("WARNING: Log directory already existed, was this inteded behaviour?")
    
        except: 
            print('Log directory did not exist, creating it at: ' + logDir)
            os.makedirs(logDir)
            
        self.logDir = logDir
        
    
    def writeToFile(self, string='', file=logTextFile, toPrint = False):

        if(toPrint):
            print(string)
        
        fd = open(self.logDir + '\\' + file, 'a') 
        fd.write(string)
        fd.close()        
                
    def writeLine(self, string='', file=logTextFile, toPrint = True): #writes a line of text in a log file

    
        if(toPrint):
            print(string)
        
        fd = open(self.logDir + '\\' + file, 'a') 
        fd.write(string + '\n')
        fd.close()
        
    def saveFile(self, data, directory='', filename='data'): #saves a file using the directory of this log object as a point of reference
        
        if(directory != ''):
            self.createDirIfNotExistent(directory)
        
        fd = open(self.logDir + '\\' + directory + '\\' + filename, 'wb') 
        pickle.dump(data, fd)
        fd.close()
    
    def saveDataframe(self, df, directory='', filename='dataframe.csv'): #saves a dataframe using this log object as a reference
        
        if(directory != ''):
            self.createDirIfNotExistent(directory)
        
        df.to_csv(self.logDir + '\\' + directory + '\\' + filename, index=False)
        
    def loadDataframe(self, directory='', filename='dataframe.csv'):
                
        return pd.read_csv(self.logDir + '\\' + directory + '\\' + filename)
        
  
    
    def createDirIfNotExistent(self, dir): #creates a dir if it does no exist
        
        dir = self.logDir + '\\' + dir
        
        try:
            os.listdir(dir)
        
        except:
            os.makedirs(dir)
            
        
    def openFile(self, fileRelativePath): #reads and returns a file
        fd = open(self.logDir + '\\' + fileRelativePath, 'rb') 
        toReturn = pickle.load(fd)
        fd.close()
        return toReturn
    
    
    def openChildLog(self, logName):
        return openLog(logDir=self.logDir, logName=logName)
    
    def createProfile(self, name : str = '', object_with_name = None):
        print("Type of object with name: " + str(type(object_with_name)) + " and name passed: " + name)
        return LoggerProfile(self, name, object_with_name)


class LoggerProfile(LogClass):
    
    class NamedObject:
        def __init__(self, name):
            self.name = name
    
    def __init__(self, lg: LogClass, name : str = '', object_with_name=None):
        
        '''Initialized a profile, that is a simple wrapper for a logger with a name
        
        Args:
            lg: The original log object
            name: The name of the object
            object_with_name: If we are to ignore the name parameter, and instead reference a logic with the .name attribute'''
        
        self.lg = lg
        
        if object_with_name == None:
            self.object_with_name = LoggerProfile.NamedObject(name)
            
        else:
            self.object_with_name = object_with_name
            
    
    def __getattr__(self, name):
        
        if name == "writeLine":
            return self.writeLine
        
        elif name == "createProfile":
            return self.createProfile
        
        else:
            return getattr(self.lg, name)

        
    def writeLine(self, string='', **params): #writes a line of text in a log file
            
        params["string"] = f'{self.object_with_name.name}: {string}'
            
        self.lg.writeLine(**params)
        
    def createProfile(self, name : str = '', object_with_name = None):
        print("Type of object with name: " + str(type(object_with_name)) + " and name passed: " + name)
        return LoggerProfile(self.lg, name, object_with_name)
     
     
def createNewLogDirIfExistent(logDir):
    
    newLogDir = logDir
        
    foundDir = False
    
    try:
        
        os.listdir(logDir)
        foundDir = True
        print("Directory " + logDir + " already existed, generating a new one...")
        
    except:
        
        foundDir = False
    
    number = 1
    
    while foundDir:
        
        newLogDir = logDir + '_' + str(number)
        
        try:
        
            os.listdir(newLogDir)
            foundDir = True
            number += 1            
        except:
        
            foundDir = False
    
    
    return newLogDir




def openLog(logDir='data\\logs', logName='', useLogName=True):
    
    """
    Creates and returns a Log object associated with a new directory in logDir/logName
    This directory can be used to do more than just writing a (text) log
    

    Args:
        logDir (str, optional): the directory, inside the project directory, where the Log directory will stay. Defaults to 'data\\logs'.
        logName (str, optional): the folder name of the Log, if empty, will be automatically generated as Log_x.

    Returns:
        LogClass: a Log object
    """
            
    print("Opening a log in directory: " + logDir + ", with name:" + logName)
    
    try:
        logsDirectories = os.listdir(logDir) #is the logDir already created? if not, error
        
    except:
        print("Logs folder did not exist, creating one at " + logDir)
        os.makedirs(logDir)
        logsDirectories = os.listdir(logDir)
    
    if(logName == ''): #if there was no specified log name, counts the number of directories that start with log_ and creates a new one with that number. For example, if there is
    
        nextLogNumber = len([l for l in logsDirectories if l.startswith('log_')]) #counts the number of log dirs that start with log_
        logName = 'log_' + str(nextLogNumber)
        
    if useLogName:
        nextLogDir = logDir + '\\' + logName
    
    else:
        nextLogDir = logDir

    
    return LogClass(logDir=nextLogDir) #returns a directory of the log folder and the file descriptor of the log text file

