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
    
    def createProfile(self, name):
        return LoggerProfile(self, name)


class LoggerProfile(LogClass):
    
    def __init__(self, lg: LogClass, name : str):
        self.lg = lg
        self.name = name
        

    def writeLine(self, string='', **params): #writes a line of text in a log file
            
        params["string"] = f'{self.name}: {string}'
            
        self.lg.writeLine(**params)
        
    def saveFile(self, **params): #saves a file using the directory of this log object as a point of reference
        
        return self.lg.saveFile(**params)
    
    def saveDataframe(self, **params): #saves a dataframe using this log object as a reference
        
        return self.lg.saveDataFrame(**params)
        
    def loadDataframe(self, **params):
        
        return self.lg.loadDataframe(dir,** params)
  
    
    def createDirIfNotExistent(self, **params): #creates a dir if it does no exist
        return self.lg.createDirIfNotExistent(**params)
            
        
    def openFile(self, **params): #reads and returns a file
        return self.lg.openFile(**params)
    
    
    def openChildLog(self, **params):
        return self.lg.openChildLog(**params)

    def createProfile(self, **params):
        return self.lg.createProfile(params)
     
     
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
            
    print("Opening a log... Log Dir: " + logDir + " Log Name:" + logName)
    
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

