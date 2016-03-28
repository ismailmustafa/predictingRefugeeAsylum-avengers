import pandas
import glob
import ntpath

def main():
	convertWeatherFiles()

def convertWeatherFiles():
	weatherFiles = glob.glob("data/clean/*.dta")
	for f in weatherFiles:
		fileName = ntpath.basename(f).split(".")[0] + ".csv"
		fFrame = pandas.read_stata(f)
		fFrame.to_csv("data/weather/" + fileName)

if __name__ == '__main__':
  main()
