import nfldb
import math
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import numpy as np

def get_score(pp):
	pts = 0.0
	pts += (pp.receiving_yds * 0.1)
	pts += (pp.receiving_tds * 7.0)
	pts += (pp.rushing_yds * 0.1)
	pts += (pp.rushing_tds * 7.0)
	pts += (pp.passing_yds * 0.04)
	pts += (pp.passing_tds * 4.0)
	pts -= (pp.fumbles_lost * 2.0)
	pts -= (pp.passing_int * 2.0)

	return pts

def get_scores(name, year):
	scores = []
	for week in range(1,17):
		q = nfldb.Query(db)
		#print 'week: %i' % week
		stats = q.player(full_name=name).game(season_year=year, season_type='Regular', week=week).as_aggregate()
		if len(stats) > 0:
			#print stats[0].receiving_yds, stats[0].receiving_tds
			score = get_score(stats[0])
			scores.append(score)
			#print score
	return scores

db = nfldb.connect()



# get set of relevant players 
#p = "John Brown"
#p = "Larry Fitzgerald"
#p = "Jordy Nelson"
p = "Julian Edelman"
#p = "Rob Gronkowski"
#p = "Tom Brady"
#p = "Drew Brees"
#p = "Jay Cutler"
#p = "Carson Palmer"
#p = "Matt Forte"
#p = "LeGarrette Blount"
#p = "Aaron Rodgers"

def get_qbs(year):
	qbs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular')
	for pp in q.sort('passing_yds').limit(12).as_aggregate():
		    #print pp.player, pp.passing_yds
		    qbs.append(pp.player.full_name)
	return qbs
def get_rbs(year):
	rbs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='RB')
	for pp in q.sort('rushing_yds').limit(24).as_aggregate():
		    #print pp.player, pp.rushing_yds
		    rbs.append(pp.player.full_name)
	return rbs
def get_wrs(year):
	wrs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='WR')
	for pp in q.sort('receiving_yds').limit(24).as_aggregate():
		    #print pp.player, pp.receiving_yds
		    wrs.append(pp.player.full_name)
	return wrs
def get_tes(year):
	tes = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='TE')
	for pp in q.sort('receiving_yds').limit(12).as_aggregate():
		    #print pp.player, pp.receiving_yds
		    tes.append(pp.player.full_name)
	return tes
		    
def get_stats(scores, name):
	tot = 0.0
	for s in scores:
		tot += s
	
	mean = tot/float(len(scores))
	#print "avg: %d"  % (mean)

	s = 0.0
	for g in scores:
		s += (g-mean) ** 2

	var = s/float(len(scores))
	#print "variance: %d" % (var)
	sd = math.sqrt(var)
	
	l_quart = np.percentile(scores, 25)
	u_quart = np.percentile(scores, 75)
	#print "sd: %d" % (sd)
	#print "games: %i" % len(scores)
	#print "avg: %d sd: %d var: %d range: %d-%d" % (mean, sd, var, mean - sd, mean + sd)
	
	return {'id': name+str(year), 'name':name, 'year': year, 'mean':mean, 'std_dev': sd, 'variance': var, 'range_low': mean-sd, 'range_high': mean+sd, 'games': len(scores), 'lquart': l_quart, 'uquart': u_quart}
	#return {'id': name+str(year), 'variance': var, 'mean': mean}


scores = []
tot = 0
features = []
for year in range(2009,2016):
	print year
	
	qbs = get_qbs(year)
	qb_scores = []
	qb_tot = 0
	qb_tot_avg = 0
	for q in qbs:
		#print q
		year_scores = get_scores(q, year)
		res = get_stats(year_scores, q)
		qb_scores.append(res)
		qb_tot += res['variance']
		qb_tot_avg += res['mean']
	#qb_scores2 = sorted(qb_scores, key=lambda player:player[1], reverse=True)
	print "avg QB variance: %d" % (qb_tot/len(qb_scores))
	print "avg QB score: %d" % (qb_tot_avg/len(qb_scores))
	
	for p in sorted(qb_scores, key=lambda player:(player['variance'],1/player['mean'])):
		print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	print "-------------------------------------------------"

	rbs = get_rbs(year)
	rb_scores = []
	rb_tot = 0
	rb_tot_avg = 0
	for r in rbs:
		#print r
		year_scores = get_scores(r, year)
		res = get_stats(year_scores, r)
		rb_scores.append(res)
		features.append(res)
		rb_tot += res['variance']
		rb_tot_avg += res['mean']
	print "avg RB variance: %d" % (rb_tot/len(rb_scores))
	print "avg RB score: %d" % (rb_tot_avg/len(rb_scores))
	#rb_scores2 = sorted(rb_scores, key=lambda player:player[1], reverse=True)
	for p in sorted(rb_scores, key=lambda player:(player['variance'],1/player['mean'])):
		print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	print "-------------------------------------------------"

	wrs = get_wrs(year)
	wr_scores = []
	wr_tot = 0
	wr_tot_avg = 0
	for w in wrs:
		#print w
		year_scores = get_scores(w, year)
		res = get_stats(year_scores, w)
		wr_scores.append(res)
		features.append(res)
		wr_tot += res['variance']
		wr_tot_avg += res['mean']
	print "avg WR variance: %d" % (wr_tot/len(wr_scores))
	print "avg WR score: %d" % (wr_tot_avg/len(wr_scores))
	#wr_scores2 = sorted(wr_scores, key=lambda player:player[1], reverse=True)
	for p in sorted(wr_scores, key=lambda player:(player['variance'],1/player['mean'])):
		print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	
	tes = get_tes(year)
	te_scores = []
	te_tot = 0
	te_tot_avg = 0
	for t in tes:
		#print w
		year_scores = get_scores(t, year)
		res = get_stats(year_scores, t)
		te_scores.append(res)
		features.append(res)
		te_tot += res['variance']
		te_tot_avg += res['mean']
	print "avg TE variance: %d" % (te_tot/len(te_scores))
	print "avg TE score: %d" % (te_tot_avg/len(te_scores))
	for p in sorted(te_scores, key=lambda player:(player['variance'],1/player['mean'])):
		print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	
	'''
	for week in range(1,17):
		q = nfldb.Query(db)
		print 'week: %i' % week
		stats = q.player(full_name=p).game(season_year=year, season_type='Regular', week=week).as_aggregate()
		if len(stats) > 0:
			print stats[0].receiving_yds, stats[0].receiving_tds
			score = get_score(stats[0])
			scores.append(score)
			print score
			tot += score
	'''
'''
k=2
vec = DictVectorizer()
data = vec.fit_transform(features)
kmeans = KMeans(n_clusters=k)
#y_pred = kmeans.fit(data)
y_pred = kmeans.fit_predict(data)

labels = kmeans.labels_
#print labels
centroids = kmeans.cluster_centers_

print y_pred
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.scatter(data[:, 0], data[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")
plt.show()

'''

'''
#print centroids
fig = plt.figure(fignum, figsize=(4, 3))

pyplot.clf()
ax = Axes2D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
est.fit(X)
labels = est.labels_

ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
fignum = fignum + 1



for i in range(k):
    # select only data observations with cluster label == i
    ds = data[np.where(labels==i)]
    # plot the data observations
    print "here:"
    print ds[0].__class__
    print "end"
    pyplot.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
pyplot.show()
'''
