'''
Recommend players based on a partial team and
plot trend line for increasing or decreasing variance.
'''

import nfldb
import math
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import numpy as np
import argparse

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
		stats = q.player(full_name=name).game(season_year=year, season_type='Regular', week=week).as_aggregate()
		if len(stats) > 0:
			score = get_score(stats[0])
			scores.append(score)
	return scores

db = nfldb.connect()
parser = argparse.ArgumentParser(description='Suggests a player to fill out a fantasy football team.')
parser.add_argument('players', metavar='player', type=str, nargs='+',
		                   help='player already picked on the team')
parser.add_argument('--needed-position', metavar='pos', type=str,
                   help='The position that needs to be filled on the team')
args = parser.parse_args()

def get_qbs(year):
	qbs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular')
	for pp in q.sort('passing_yds').limit(24).as_aggregate():
		    qbs.append(pp.player.full_name)
	return qbs
def get_rbs(year):
	rbs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='RB')
	for pp in q.sort('rushing_yds').limit(36).as_aggregate():
		    rbs.append(pp.player.full_name)
	return rbs
def get_wrs(year):
	wrs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='WR')
	for pp in q.sort('receiving_yds').limit(48).as_aggregate():
		    wrs.append(pp.player.full_name)
	return wrs
def get_tes(year):
	tes = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='TE')
	for pp in q.sort('receiving_yds').limit(24).as_aggregate():
		    tes.append(pp.player.full_name)
	return tes
		    
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


scores = []
tot = 0
features = []
stored_data = {}
all_avgs = {}
score_map = {}
for year in range(2014,2015):
	all_avgs['qb'] = []
	qbs = get_qbs(year)
	qb_scores = []
	qb_tot = 0
	qb_tot_avg = 0
	for q in qbs:
		year_scores = get_scores(q, year)
		res = get_stats(year_scores, q)
		qb_scores.append(res)
		if res['games'] > 10:
			stored_data[res['id']] = res
			all_avgs['qb'].append(res)
		score_map[res['id']] = year_scores
		qb_tot += res['variance']
		qb_tot_avg += res['mean']

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
		if res['games'] > 10:
			stored_data[res['id']] = res
			all_avgs['rb'].append(res)
		score_map[res['id']] = year_scores
		rb_tot += res['variance']
		rb_tot_avg += res['mean']

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
		if res['games'] > 10:
			stored_data[res['id']] = res
			all_avgs['wr'].append(res)
		score_map[res['id']] = year_scores
		wr_tot += res['variance']
		wr_tot_avg += res['mean']
	
	all_avgs['te'] = []
	tes = get_tes(year)
	te_scores = []
	te_tot = 0
	te_tot_avg = 0
	for t in tes:
		year_scores = get_scores(t, year)
		res = get_stats(year_scores, t)
		te_scores.append(res)
		if res['games'] > 10:
			stored_data[res['id']] = res
			all_avgs['te'].append(res)
		score_map[res['id']] = year_scores
		te_tot += res['variance']
		te_tot_avg += res['mean']
	
avg_score = 82
min_score = avg_score - 0.5
max_score = avg_score + 0.5

def get_score(team, week):
	tot = 0
	for p in team:
		tot += score_map[p][week]
	return tot
def calc_avg_mean(team):
	tot = 0.0
	for p in team:
		tot += stored_data[p]['mean']
	return tot
def calc_avg_variance(team):
	tot = 0.0
	for p in team:
		tot += stored_data[p]['variance']
	return tot/len(team)


curr_team = [ a+'2014' for a in args.players] 
print "TEAM:"
print args.players
needed_position = args.needed_position
print "Looking for:"
print needed_position
pteams = []
for p in all_avgs[needed_position]:
	new_team = [p['id'],] +curr_team
	data = {"team":new_team, 'player':p['id'], 'variance':calc_avg_variance(new_team), 'wins':0,'losses':0}
	for p_new in all_avgs[needed_position]:
		if p_new['id'] not in new_team:
			t2 = [p_new['id'],] + curr_team
			for week in range(10):
				t1_score = get_score(new_team,week)
				t2_score = get_score(t2,week)
				if t1_score >= t2_score:
					data['wins'] += 1
				else:
					data['losses'] += 1
	pteams.append(data)


pts = [[],[]]
for t in pteams:
	pts[0].append(t['wins'])
	pts[1].append(t['variance'])

suggestions = 10
print "Suggestions:"
for p in sorted(pteams, key=lambda team:1.0/team['wins']):
		if suggestions >0:
			suggestions -= 1
			print '-' * 30
			print "%i-%i(%f)(%s)" % (p['wins'], p['losses'], p['variance'],p['player'])

k = np.poly1d(np.polyfit(pts[1], pts[0], 1))
print k

plt.plot(pts[1], pts[0], 'ro')
plt.plot(pts[1], np.poly1d(np.polyfit(pts[1], pts[0], 1))(pts[1]))
plt.title('Wins vs Variance')
plt.ylabel('Wins')
plt.xlabel('Variance')
plt.grid(True)
plt.show()
