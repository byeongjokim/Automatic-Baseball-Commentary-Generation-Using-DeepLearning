import rdflib

def search_gameInfo(gameCode, inn, score, gameinfo):
	homescore = score[0]
	awayscore = score[1]

	stadium = gameinfo[0]
	date = gameinfo[1]

	annotation = [
		str(date) + " " + str(stadium) + "에서 경기 진행중입니다.",
		str(date) + "에 진행하는 경기 현재 경기 스코어 "+str(homescore)+" 대 "+str(awayscore)+" 입니다.",
		#str(inn).split("_")[-1] + " 경기 스코어 "+str(homescore)+" 대 "+str(awayscore)+" 입니다.",
		str(inn).split("_")[-1] + " 경기 스코어 "+str(homescore)+" 대 "+str(awayscore)+" 진행중입니다."

	]
	return annotation

def search_pitcher(gameCode, p):
	g = rdflib.Graph()
	g.load('_data/_owl/180515SKOB.owl')
	uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

	inGame = rdflib.URIRef(uri + "inGame")
	thisGame = rdflib.URIRef(uri + gameCode)

	fromPitcher = rdflib.URIRef(uri + "fromPitcher")
	pitcher = rdflib.URIRef(uri + p)
	thisERA = rdflib.URIRef(uri + "thisERA")

	result = rdflib.URIRef(uri + "result")

	query = "SELECT ?o where {?pitcher ?thisERA ?o}"
	r = g.query(query, initBindings={"pitcher": pitcher, "thisERA": thisERA})

	era = 0
	for row in r:
		era = row[0]

	query = "SELECT ?o where {?s ?fromPitcher ?pitcher . ?s ?inGame ?thisGame . ?s ?result ?o}"
	r = g.query(query, initBindings={"fromPitcher": fromPitcher, "pitcher": pitcher,
									 "inGame": inGame, "thisGame": thisGame,
									 "result": result})
	strikeout = 0
	baseonballs = 0

	for row in r:
		if ("Strikeout" in row[0]):
			strikeout = strikeout + 1
		if ("BaseOnBalls" in row[0]):
			baseonballs = baseonballs + 1

	annotation = [
		p + " 투수 오늘 경기 "+str(len(r))+"번째 타석에서 공을 던지고 있습니다.",
		p + " 투수 오늘 경기 "+str(strikeout)+"개의 삼진을 잡아내고 있습니다.",
		p + " 투수 오늘 경기 "+str(baseonballs)+"개의 포볼로 타자를 진루 시켰습니다.",
		p + " 투수 이번 시즌 "+str(era)+"의 평균 자책점을 기록하고 있습니다.",
		p + " 투수 과연 어떤 공을 던질까요?",
	]

	return annotation

def search_batter(gameCode, b):
	g = rdflib.Graph()
	g.load('_data/_owl/180515SKOB.owl')
	uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

	inGame = rdflib.URIRef(uri + "inGame")
	thisGame = rdflib.URIRef(uri + gameCode)

	toHitter = rdflib.URIRef(uri + "toHitter")
	batter = rdflib.URIRef(uri + b)
	thisAVG = rdflib.URIRef(uri + "thisAVG")

	result = rdflib.URIRef(uri + "result")
	stayIn1stBase = rdflib.URIRef(uri + "stayIn1stBase")

	query = "SELECT ?o where {?batter ?thisAVG ?o}"
	r = g.query(query, initBindings={"batter": batter, "thisAVG": thisAVG})

	avg = 0
	for row in r:
		avg = row[0]

	query = "SELECT ?o where {?s ?toHitter ?hitter . ?s ?inGame ?thisGame . ?s ?result ?o } order by ?s"
	r = g.query(query, initBindings={"toHitter": toHitter, "hitter": batter, "inGame": inGame, "thisGame": thisGame, "result": result})
	this_game_count = len(r)
	batter_history = []
	for row in r:
		batter_history.append(row[0].split("#")[1].split("_")[1])

	query = "SELECT ?o where {?s ?toHitter ?hitter . ?s ?result ?o . ?s ?stayIn1stBase ?o1} order by ?s"
	r = g.query(query, initBindings={"toHitter": toHitter, "hitter": batter,
									 "inGame": inGame, "thisGame": thisGame,
									 "result": result, "stayIn1stBase": stayIn1stBase})

	batter_history_when1st = []
	for row in r:
		batter_history_when1st.append(row[0].split("#")[1].split("_")[1])

	annotation = [
		b + " 타자의 오늘 " + str(this_game_count+1) + "번째 타석입니다.",
		b + " 타자는 이번 시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다.",
		b + " 타자 이번 타석 안타를 기록 할 수 있을까요?"
	]

	if (batter_history):
		annotation.append(b + " 타자 오늘 " + str(this_game_count+1) + "번째 타석입니다, " + ", ".join(_ for _ in batter_history) + "을 기록하고 있습니다.")
		annotation.append(b + " 타자 저번 타석, " + str(batter_history[-1]) + "을 기록하였습니다.")
	if (batter_history_when1st):
		annotation.append(b + " 타자 1루 주자가 있는 상황에서 " + ", ".join(_ for _ in batter_history_when1st) + "을 기록하고 있습니다.")
		annotation.append("오늘 1루 주자가 있는 타석에서 " + b + " 타자 최근, " + str(batter_history_when1st[-1]) + "을 기록하였습니다.")

	return annotation

def search_pitcherbatter(gameCode, p, b):
	g = rdflib.Graph()
	g.load('_data/_owl/180515SKOB.owl')
	uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

	inGame = rdflib.URIRef(uri + "inGame")
	thisGame = rdflib.URIRef(uri + gameCode)

	toHitter = rdflib.URIRef(uri + "toHitter")
	batter = rdflib.URIRef(uri + b)
	thisAVG = rdflib.URIRef(uri + "thisAVG")

	fromPitcher = rdflib.URIRef(uri + "fromPitcher")
	pitcher = rdflib.URIRef(uri + p)
	thisERA = rdflib.URIRef(uri + "thisERA")

	result = rdflib.URIRef(uri + "result")

	avg = 0
	era = 0
	query = "SELECT ?o where {?s ?p ?o}"
	r = g.query(query, initBindings={"s":batter, "p": thisAVG})
	for row in r:
		avg = row[0]

	r = g.query(query, initBindings={"s": pitcher, "p": thisERA})
	for row in r:
		era = row[0]

	query = "SELECT ?o where {?s ?inGame ?thisGame . ?s ?toHitter ?hitter . ?s ?fromPitcher ?pitcher . ?s ?result ?o} order by desc(?s)"
	r = g.query(query, initBindings={
		"inGame": inGame, "thisGame": thisGame,
		"toHitter": toHitter, "hitter": batter,
		"fromPitcher": fromPitcher, "pitcher": pitcher,
		"result": result
	})

	result_history = []
	strikeout = 0
	getonbase = 0
	for row in r:
		result_history.append(row[0].split("#")[1].split("_")[1])

		if ("Strikeout" in row[0]):
			strikeout = strikeout + 1
		if ("BaseOnBalls" in row[0]):
			getonbase = getonbase + 1
		if ("HitByPitch" in row[0]):
			getonbase = getonbase + 1
		if ("Double" in row[0]):
			getonbase = getonbase + 1
		if ("HomeRun" in row[0]):
			getonbase = getonbase + 1
		if ("Triple" in row[0]):
			getonbase = getonbase + 1
		if ("SingleHit" in row[0]):
			getonbase = getonbase + 1

	annotation = [
		b + " 타자 " + p + " 투수의 신경전 속에 " + b + " 타자는 이번시즌 " + str(avg) + "의 펑균 타율을 기록하고 있습니다.",
		b + " 타자 " + p + " 투수의 신경전 속에 " + p + " 투수는 이번시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다.",
		b + " 타자 " + p + " 투수를 상대로 오늘 " + ", ".join(_ for _ in result_history) + "을 기록 하였습니다.",
		b + " 타자 " + p + " 투수를 상대로 오늘 " + str(getonbase) + "개의 안타 기록 하였습니다.",
		p + " 투수 " + b + " 타자를 상대로 오늘 경기 " + str(getonbase) + "개의 안타를 허용 하였습니다.",
		p + " 투수 " + b + " 타자를 상대로 오늘 경기 " + str(strikeout) + "개의 스트라이크 아웃을 잡아냈습니다.",
		"투수와 타자 사이에 팽팽한 긴장감이 감지됩니다.",
	]
	return annotation

def search_runner(gameCode):
	return 1
"""
search_pitcher("20180515OBSK", "후랭코프68240")
search_batter("20180515OBSK", "한동민62895")
search_pitcherbatter("20180515OBSK", "후랭코프68240", "한동민62895")
"""