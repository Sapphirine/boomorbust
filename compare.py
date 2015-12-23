'''
Script for comparing teams and determining optimal levels of variance

'''

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
		stats = q.player(full_name=name).game(season_year=year, season_type='Regular', week=week).as_aggregate()
		if len(stats) > 0:
			score = get_score(stats[0])
			scores.append(score)
	return scores

db = nfldb.connect()

def get_qbs(year):
	qbs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular')
	for pp in q.sort('passing_yds').limit(12).as_aggregate():
		    qbs.append(pp.player.full_name)
	return qbs
def get_rbs(year):
	rbs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='RB')
	for pp in q.sort('rushing_yds').limit(24).as_aggregate():
		    rbs.append(pp.player.full_name)
	return rbs
def get_wrs(year):
	wrs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='WR')
	for pp in q.sort('receiving_yds').limit(36).as_aggregate():
		    wrs.append(pp.player.full_name)
	return wrs
def get_tes(year):
	tes = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='TE')
	for pp in q.sort('receiving_yds').limit(12).as_aggregate():
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
for year in range(2009,2016):
	print year
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
	print "avg QB variance: %d" % (qb_tot/len(qb_scores))
	print "avg QB score: %d" % (qb_tot_avg/len(qb_scores))
	
	for p in sorted(qb_scores, key=lambda player:(player['variance'],1/player['mean'])):
		print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	print "-------------------------------------------------"

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
	print "avg RB variance: %d" % (rb_tot/len(rb_scores))
	print "avg RB score: %d" % (rb_tot_avg/len(rb_scores))
	#rb_scores2 = sorted(rb_scores, key=lambda player:player[1], reverse=True)
	for p in sorted(rb_scores, key=lambda player:(player['variance'],1/player['mean'])):
		print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	print "-------------------------------------------------"

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
	print "avg WR variance: %d" % (wr_tot/len(wr_scores))
	print "avg WR score: %d" % (wr_tot_avg/len(wr_scores))
	#wr_scores2 = sorted(wr_scores, key=lambda player:player[1], reverse=True)
	for p in sorted(wr_scores, key=lambda player:(player['variance'],1/player['mean'])):
		print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	
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


avg_score = 82
min_score = avg_score - 0.5
max_score = avg_score + 0.5


teams = []
tgt = 50000
def get_players(pts, team, needed_pos, tried, players):
	if len(needed_pos) > 0 and len(teams) < tgt:
		ret = []
		new_needed = []
		if len(needed_pos) > 1:
			new_needed = needed_pos[1:]
		local_tried = set([])
		for p in players[needed_pos[0]]:
			if p['id'] not in team and (pts + p['mean'] <= max_score) and p['id'] not in tried:
				local_tried.add(p['id'])
				get_players(p['mean'] + pts, set([p['id'],]) | team, new_needed, tried|local_tried,players) 	
	else:
		if pts >= min_score and pts <= max_score and len(team) == 7:
			teams.append(team)
	return tried
print all_avgs.keys()
print len(all_avgs['qb'])
tried = get_players(0,set([]),['qb','rb','rb','wr','wr','wr','te'],set([]), all_avgs)

print teams
print "tried: %i" % len(tried)


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
record = [{} for k in range(len(teams))]
for i,t in enumerate(teams):
	record[i] = {'wins':0,'losses':0, 'team':t}
	record[i]['variance'] = calc_avg_variance(t)
	record[i]['mean'] = calc_avg_mean(t)
	for t2 in teams:
		if t == t2:
			continue
		for week in range(10):
			t1_score = get_score(t,week)
			t2_score = get_score(t2,week)
			if t1_score >= t2_score:
				record[i]['wins'] += 1
			else:
				record[i]['losses'] += 1

for p in sorted(record, key=lambda team:(team['wins'],1/team['mean'])):
		print '-' * 30
		print "Team:"
		print "%i-%i(%f)(%f)" % (p['wins'], p['losses'], p['variance'], p['mean'])
		print p['team']


pts = [[],[]]
for t in record:
	pts[0].append(t['wins'])
	pts[1].append(t['variance'])

plt.plot(pts[1], pts[0], 'ro')
plt.plot(pts[1], np.poly1d(np.polyfit(pts[1], pts[0], 1))(pts[1]))
plt.title('Wins vs Variance')
plt.ylabel('Wins')
plt.xlabel('Variance')
plt.grid(True)
plt.show()

