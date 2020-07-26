## arena_melon

- traces made for the 3rd kakao arena challenge (melon playlist continuation) [star]
- could be updated to verify other hypotheses even after the end of the official event :)
- the final model submitted is basically based on MF(ALS), and CB model for cold-start cases
 > first of all, popularity-based item filterting was done (only 30% of total items was used)
 > for cold-start cases, playlist titles were exploited to make comparison and calculate similarities among them
 > for non-cold cases, MF(ALS) method was applied thanks to _implicit_ library, with a little hyper-parameter tuning
 > for tag prediction, some heuristic approach was made
   : calculate the tags per each song and again aggregate the tags of the songs in each playlist
   
- regarding the task as a kind of 'next-item recommendation' problem, several deep learning based approaches were tried..
 : but some showed worse performances than MF, while others were found to hard to optimize for now (will fix soon)
 
- happy to have a tons of trials & errors in order to (come up with ideas) & (make it work) 
- and most of all, many thanks to kakao for this opportunity :)
