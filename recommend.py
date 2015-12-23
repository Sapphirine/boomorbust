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
		#print 'week: %i' % week
		stats = q.player(full_name=name).game(season_year=year, season_type='Regular', week=week).as_aggregate()
		if len(stats) > 0:
			#print stats[0].receiving_yds, stats[0].receiving_tds
			score = get_score(stats[0])
			scores.append(score)
			#print score
	return scores

db = nfldb.connect()
parser = argparse.ArgumentParser(description='Suggests a player to fill out a fantasy football team.')
parser.add_argument('players', metavar='player', type=str, nargs='+',
		                   help='player already picked on the team')
parser.add_argument('--needed-position', metavar='pos', type=str,
                   help='The position that needs to be filled on the team')
args = parser.parse_args()


# get set of relevant players 
#p = "John Brown"
#p = "Larry Fitzgerald"
#p = "Jordy Nelson"
#p = "Julian Edelman"
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
	for pp in q.sort('passing_yds').limit(24).as_aggregate():
		    #print pp.player, pp.passing_yds
		    qbs.append(pp.player.full_name)
	return qbs
def get_rbs(year):
	rbs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='RB')
	for pp in q.sort('rushing_yds').limit(36).as_aggregate():
		    #print pp.player, pp.rushing_yds
		    rbs.append(pp.player.full_name)
	return rbs
def get_wrs(year):
	wrs = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='WR')
	for pp in q.sort('receiving_yds').limit(48).as_aggregate():
		    #print pp.player, pp.receiving_yds
		    wrs.append(pp.player.full_name)
	return wrs
def get_tes(year):
	tes = []
	q = nfldb.Query(db)
	q.game(season_year=year, season_type='Regular').player(position='TE')
	for pp in q.sort('receiving_yds').limit(24).as_aggregate():
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
stored_data = {}
all_avgs = {}
score_map = {}
for year in range(2014,2015):
	#print year
	all_avgs['qb'] = []
	qbs = get_qbs(year)
	qb_scores = []
	qb_tot = 0
	qb_tot_avg = 0
	for q in qbs:
		#print q
		year_scores = get_scores(q, year)
		res = get_stats(year_scores, q)
		qb_scores.append(res)
		if res['games'] > 10:
			stored_data[res['id']] = res
			all_avgs['qb'].append(res)
		score_map[res['id']] = year_scores
		qb_tot += res['variance']
		qb_tot_avg += res['mean']
	#qb_scores2 = sorted(qb_scores, key=lambda player:player[1], reverse=True)
	#print "avg QB variance: %d" % (qb_tot/len(qb_scores))
	#print "avg QB score: %d" % (qb_tot_avg/len(qb_scores))
	
	#for p in sorted(qb_scores, key=lambda player:(player['variance'],1/player['mean'])):
	#	print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	#print "-------------------------------------------------"

	all_avgs['rb'] = []
	rbs = get_rbs(year)
	rb_scores = []
	rb_tot = 0
	rb_tot_avg = 0
	for r in rbs:
		#print r
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
	#print "avg RB variance: %d" % (rb_tot/len(rb_scores))
	#print "avg RB score: %d" % (rb_tot_avg/len(rb_scores))
	#rb_scores2 = sorted(rb_scores, key=lambda player:player[1], reverse=True)
	#for p in sorted(rb_scores, key=lambda player:(player['variance'],1/player['mean'])):
	#	print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	#print "-------------------------------------------------"

	all_avgs['wr'] = []
	wrs = get_wrs(year)
	wr_scores = []
	wr_tot = 0
	wr_tot_avg = 0
	for w in wrs:
		#print w
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
	#print "avg WR variance: %d" % (wr_tot/len(wr_scores))
	#print "avg WR score: %d" % (wr_tot_avg/len(wr_scores))
	#wr_scores2 = sorted(wr_scores, key=lambda player:player[1], reverse=True)
	#for p in sorted(wr_scores, key=lambda player:(player['variance'],1/player['mean'])):
	#	print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	
	all_avgs['te'] = []
	tes = get_tes(year)
	te_scores = []
	te_tot = 0
	te_tot_avg = 0
	for t in tes:
		#print w
		year_scores = get_scores(t, year)
		res = get_stats(year_scores, t)
		te_scores.append(res)
		if res['games'] > 10:
			stored_data[res['id']] = res
			all_avgs['te'].append(res)
		score_map[res['id']] = year_scores
		te_tot += res['variance']
		te_tot_avg += res['mean']
	#print "avg TE variance: %d" % (te_tot/len(te_scores))
	#print "avg TE score: %d" % (te_tot_avg/len(te_scores))
	#for p in sorted(te_scores, key=lambda player:(player['variance'],1/player['mean'])):
	#	print "%s avg: %d sd: %d var: %d games: %i range: %d-%d quarts: %d-%d" % (p['id'], p['mean'], p['std_dev'], p['variance'], p['games'], p['range_low'], p['range_high'], p['lquart'], p['uquart'])
	
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
#print "avg:"
#print "qb: %f" % (sum(all_avg['qb'])/len(all_avg['qb']))
#print "rb: %f" % (sum(all_avg['rb'])/len(all_avg['rb']))
#print "wr: %f" % (sum(all_avg['wr'])/len(all_avg['wr']))
#print "te: %f" % (sum(all_avg['te'])/len(all_avg['te']))


avg_score = 82
min_score = avg_score - 0.5
max_score = avg_score + 0.5

#pts -> list of players



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


curr_team = [ a+'2014' for a in args.players]#['Tom Brady2014','Matt Forte2014', 'Mike Evans2014','Antonio Brown2014','Jason Witten2014','Julian Edelman2014'] 
print "TEAM:"
print args.players
needed_position = args.needed_position#'rb'
print "Looking for:"
print needed_position
pteams = []
for p in all_avgs[needed_position]:
	new_team = [p['id'],] +curr_team
	data = {"team":new_team, 'player':p['id'], 'variance':calc_avg_variance(new_team), 'wins':0,'losses':0}
	for p_new in all_avgs[needed_position]:
		if p_new['id'] not in new_team:
			t2 = [p_new['id'],] + curr_team
			#get_players(p['mean'] + pts, set([p['id'],]) | team, new_needed, tried|local_tried,players) 	
			for week in range(10):
				t1_score = get_score(new_team,week)
				t2_score = get_score(t2,week)
				if t1_score >= t2_score:
					data['wins'] += 1
				else:
					data['losses'] += 1
	pteams.append(data)


'''
def get_players(pts, team, needed_pos, tried, players):
	if len(needed_pos) > 0 and len(teams) < tgt:
		#print "getting player: %s" % needed_pos[0]
		#print "needed:"
		#print needed_pos
		ret = []
		new_needed = []
		if len(needed_pos) > 1:
			new_needed = needed_pos[1:]
		#print len(players[needed_pos[0]])
		local_tried = set([])
		for p in players[needed_pos[0]]:
			if p['id'] not in team and (pts + p['mean'] <= max_score) and p['id'] not in tried:
				local_tried.add(p['id'])
				get_players(p['mean'] + pts, set([p['id'],]) | team, new_needed, tried|local_tried,players) 	
	else:
		#print "final:"
		#print pts
		#print team
		if pts >= min_score and pts <= max_score and len(team) == 7:
			#print "FOUND"
			teams.append(team)
	return tried
print all_avgs.keys()
print len(all_avgs['qb'])
tried = get_players(0,set([]),['qb','rb','rb','wr','wr','wr','te'],set([]), all_avgs)

print teams
print "tried: %i" % len(tried)


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
'''


pts = [[],[]]
for t in pteams:#record:
	pts[0].append(t['wins'])
	pts[1].append(t['variance'])

#fig = plt.figure(figsize=(24, 9))
#colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#ax = fig.add_subplot(1, 1, 1)
#kmeans.fit(data)
#y_pred = kmeans.fit_predict(data)
#print data
#data_arr = data.toarray()

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

'''
for k, col in zip(range(n_clusters), colors):
    my_members = labels == k
    cluster_center = centroids[k]
    print my_members
    print "-" * 25
    #print data.toarray()[my_members,0]
    print "-" * 25
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
#plt.show()


#print record

'''

'''
team = []
for q in get_players(qb):
	new_team = team + q
	for rb1 in get_players(rb, new score)
		

for qb in qb_list:
	pts = qb.points
	team = get_rbs(pts
	rbs = get rb pair(curr points)
	wrs = get wr three (curr points)
	te = check for valid te(curr points)
	if te (aka valid team):
		add as team
	
	for rb in rbs
		for rb2 in rbs+1
			for w




def total_value(items, max_weight):
    return  sum([x[2] for x in items]) if sum([x[1] for x in items]) < max_weight else 0
 
cache = {}
def solve(items, max_weight):
    if not items:
        return ()
    if (items,max_weight) not in cache:
        head = items[0]
        tail = items[1:]
        include = (head,) + solve(tail, max_weight - head[1])
        dont_include = solve(tail, max_weight)
        if total_value(include, max_weight) > total_value(dont_include, max_weight):
            answer = include
        else:
            answer = dont_include
        cache[(items,max_weight)] = answer
    return cache[(items,max_weight)]
'''

'''
n_clusters=2
vec = DictVectorizer()
data = vec.fit_transform(features)
kmeans = KMeans(n_clusters=n_clusters)

kmeans.fit(data)
#y_pred = kmeans.fit_predict(data)

labels = kmeans.labels_
#print labels
centroids = kmeans.cluster_centers_

case = stored_data['Mike Evans2014']
print "case:"
print case
case_vec = vec.fit_transform({'variance':case['variance'],'mean':case['mean']})
case_res = kmeans.predict(case_vec)
print "res:"
print case_res

case = stored_data['LeSean McCoy2015']
print "case:"
print case
case_vec = vec.fit_transform({'variance':case['variance'],'mean':case['mean']})
case_res = kmeans.predict(case_vec)
print "res:"
print case_res
case = stored_data['DeAngelo Williams2015']
print "case:"
print case
case_vec = vec.fit_transform({'variance':case['variance'],'mean':case['mean']})
case_res = kmeans.predict(case_vec)
print "res:"
print case_res
'''
'''
kmeans.fit(data)
#y_pred = kmeans.fit_predict(data)

labels = kmeans.labels_
#print labels
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(24, 9))
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
ax = fig.add_subplot(1, 1, 1)
kmeans.fit(data)
#y_pred = kmeans.fit_predict(data)
print data
data_arr = data.toarray()
for k, col in zip(range(n_clusters), colors):
    my_members = labels == k
    cluster_center = centroids[k]
    print my_members
    print "-" * 25
    #print data.toarray()[my_members,0]
    print "-" * 25
    ax.plot(data_arr[my_members, 0], data_arr[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show()

'''

'''

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
x = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
plt.show()
'''
'''
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
