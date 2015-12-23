import nfldb
import math
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import numpy as np
import argparse


db = nfldb.connect()

# parse input arguments
parser = argparse.ArgumentParser(description='Classify a player as "Boom or Bust" or "Consistent".')
parser.add_argument('--player', metavar='player', type=str,
                   help='The player to be analyzed')
parser.add_argument('--year', metavar='year', type=int,
                   help='The year to gather data from')
args = parser.parse_args()

# Determine the fantasy score for an individual game
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

# Determine the fantasy scores for each game in a season
def get_scores(name, year):
	scores = []
	for week in range(1,17):
		q = nfldb.Query(db)
		stats = q.player(full_name=name).game(season_year=year, season_type='Regular', week=week).as_aggregate()
		if len(stats) > 0:
			score = get_score(stats[0])
			scores.append(score)
	return scores

# Returns a list of the fantasy relevant QBs for a specified year
def get_qbs(year):
	qbs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular')
	for pp in q.sort('passing_yds').limit(12).as_aggregate():
		    qbs.append(pp.player.full_name)
	return qbs
# Returns a list of the fantasy relevant RBs for a specified year
def get_rbs(year):
	rbs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='RB')
	for pp in q.sort('rushing_yds').limit(24).as_aggregate():
		    rbs.append(pp.player.full_name)
	return rbs
# Returns a list of the fantasy relevant WRs for a specified year
def get_wrs(year):
	wrs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='WR')
	for pp in q.sort('receiving_yds').limit(36).as_aggregate():
		    wrs.append(pp.player.full_name)
	return wrs
# Returns a list of the fantasy relevant TEs for a specified year
def get_tes(year):
	tes = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='TE')
	for pp in q.sort('receiving_yds').limit(12).as_aggregate():
		    tes.append(pp.player.full_name)
	return tes
		    
# Calculate various statistics on the list of fantasy scores for a specified player
def get_stats(scores, name):
	tot = 0.0
	for s in scores:
		tot += s
	
	mean = tot/float(len(scores))

	s = 0.0
	for g in scores:
		s += (g-mean) ** 2

	var = s/float(len(scores))
	sd = math.sqrt(var)
	
	l_quart = np.percentile(scores, 25)
	u_quart = np.percentile(scores, 75)
	
	return {'id': name+str(year), 'name':name, 'year': year, 'mean':mean, 'std_dev': sd, 'variance': var, 'range_low': mean-sd, 'range_high': mean+sd, 'games': len(scores), 'lquart': l_quart, 'uquart': u_quart}



'''
Determine the list of fantasy relevant players for each year and compute the statistics on each player
'''
scores = []
tot = 0
features = []
stored_data = {}
all_avgs = {}
for year in range(2013,2016):
	all_avgs['qb'] = []
	qbs = get_qbs(year)
	qb_scores = []
	qb_tot = 0
	qb_tot_avg = 0
	for q in qbs:
		year_scores = get_scores(q, year)
		res = get_stats(year_scores, q)
		qb_scores.append(res)
		qb_tot += res['variance']
		stored_data[res['id']] = res
		qb_tot_avg += res['mean']
		all_avgs['qb'].append(res['mean'])

	all_avgs['rb'] = []
	rbs = get_rbs(year)
	rb_scores = []
	rb_tot = 0
	rb_tot_avg = 0
	for r in rbs:
		year_scores = get_scores(r, year)
		res = get_stats(year_scores, r)
		rb_scores.append(res)
		features.append({'variance': res['variance'], 'mean':res['mean']})
		stored_data[res['id']] = res
		rb_tot += res['variance']
		rb_tot_avg += res['mean']
		all_avgs['rb'].append(res['mean'])

	all_avgs['wr'] = []
	wrs = get_wrs(year)
	wr_scores = []
	wr_tot = 0
	wr_tot_avg = 0
	for w in wrs:
		year_scores = get_scores(w, year)
		res = get_stats(year_scores, w)
		wr_scores.append(res)
		features.append({'variance': res['variance'], 'mean':res['mean']})
		stored_data[res['id']] = res
		wr_tot += res['variance']
		wr_tot_avg += res['mean']
		all_avgs['wr'].append(res['mean'])
	
	all_avgs['te'] = []
	tes = get_tes(year)
	te_scores = []
	te_tot = 0
	te_tot_avg = 0
	for t in tes:
		year_scores = get_scores(t, year)
		res = get_stats(year_scores, t)
		te_scores.append(res)
		stored_data[res['id']] = res
		te_tot += res['variance']
		te_tot_avg += res['mean']
		all_avgs['te'].append(res['mean'])

# Convert the feature dictionary to vectors expected by the KMeans algorithm
n_clusters=2
vec = DictVectorizer()
data = vec.fit_transform(features)

# Perform the KMeans clustering
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
if centroids [0][1] > centroids [1][1]:
	mapping = ['Boom or Bust','Consistent']
else:
	mapping = ['Consistent','Boom or Bust']

# Pass the input player to the model
if args.player and args.year:
	player_str = args.player+str(args.year)
	if player_str in stored_data:
		player_data = stored_data[player_str]
		player_vec = vec.fit_transform({'variance':player_data['variance'],'mean':player_data['mean']})
		clust = kmeans.predict(player_vec)
		print args.player
		print mapping[clust[0]]
		print "avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (player_data['mean'], player_data['std_dev'], player_data['variance'], player_data['games'], player_data['range_low'], player_data['range_high'], player_data['lquart'], player_data['uquart'])
	else:
		print "Player %s in year %i not found!" % (args.player, args.year)
