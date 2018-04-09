# this is a short temporary scipt python code 
# the aim of which is for DOWNLOAD some file of ruled constantly 
import urllib.request
for i in range(0, 33):
	try:
		file_name = 'http://www.tcm.phy.cam.ac.uk/~bds10/aqp/lec'+str(i)+'_compressed.pdf'
		def main():
			download_file(file_name)
		def download_file(download_url):
			response = urllib.request.urlopen(download_url)
			file = open(str(i)+".pdf", 'wb')
			file.write(response.read())
			file.close()
			print("Completed"+str(i))
		main()
	except:
		pass
