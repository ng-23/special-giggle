'''
The aim of this script is to preprocess the multi-band images from Sentinel-2 
such that they can provide more relevant/useful information in downstream applications. 
Mainly, this entails combining the various bands and calculating spectral indices to produce composite images that better emphasize certain features.

This script generates an output file - specifically, an HDF5 file. 
It is structured with each location as the top-most group, and every day (0-729) as a subgroup in each location. 
The composite images are then stored in the day subgroup of each location group.

Helpful Resources:
- [Sentinel Hub Scripts](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel/sentinel-2/)
- [Sentinel 2 Bands and Combinations](https://gisgeography.com/sentinel-2-bands-combinations/)
- [Spectral Signature Cheatsheet in Remote Sensing](https://gisgeography.com/spectral-signature/)
- [How to Make Outstanding Maps with Sentinel-2 and ArcGIS Pro - Part 1: Band Combinations](https://www.staridasgeography.gr/how-to-make-outstanding-maps-with-sentinel-2-and-arcgis-pro-part-1-band-combinations/)
- [How to Make Outstanding Maps with Sentinel-2 and ArcGIS Pro - Part 2: Spectral Indices](https://www.staridasgeography.gr/how-to-make-outstanding-maps-with-sentinel-2-and-arcgis-pro-part-2-spectral-indices/)
- [Awesome Spectral Indices](https://awesome-ee-spectral-indices.readthedocs.io/en/latest/list.html)
- [Index DataBase](https://www.indexdatabase.de/db/i.php)
'''