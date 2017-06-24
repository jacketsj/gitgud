import re
import random
import io
from threading import Thread

import requests
import json


class CSVData:

    def __init__(self, msg="", id="", stars="", forks="", watchers=""):
        self.msg = str(msg);
        self.id = str(id);
        self.stars = str(stars);
        self.forks = str(forks);
        self.watchers = str(watchers);

    def getCSVRepresentation(self):
        return self.msg + ", " + self.id + ", " + self.stars + ", " + self.forks + ", " + self.watchers;


proxies = []
userAgents = []

def loadProxies():
    global proxies
    with open("proxies.txt","r") as f:
        lines = f.readlines();
        for line in lines:
           proxies.append(line.strip());


def loadUserAgents():
    global userAgents
    with open("user-agents.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            userAgents.append(line.strip());

def randomUserAgent():
    headers = {
        "User-Agent": random.choice(userAgents)
    }

    return headers;


def randomProxy():
    while (True):
        proxy = {}
        proxy["https"] = "https://" + random.choice(proxies);
        try:
            r = requests.get("https://myexternalip.com/raw", proxies=proxy, headers=randomUserAgent(), timeout=3);
            break;
        except Exception as e:
            continue;

    print("choosing proxy: ", proxy)
    return proxy;


loadProxies()
loadUserAgents()

THREAD_COUNT = 20;

allUsersRepo = ["https://api.github.com/users/torvalds/repos"]
#commit message, repo-id, stars, forks, watchers, additions, deletions



def fetchData(startPageClean, repoData, i):
    print("scraping commit page: " + startPageClean)
    proxy = randomProxy()
    headers = randomUserAgent()
    try:
        moreCommits = requests.get(startPageClean, proxies=proxy, headers=headers, timeout=5);
        commit_data_json = moreCommits.json();
        totalCommitData = []
        for commit_data in commit_data_json:
            curmsg = commit_data["commit"]["message"].replace('\n', ' ').replace('\r', '').replace(",", "");
            curstars = repoData["stargazers_count"]
            curforks=repoData["forks"]
            curwatchers=repoData["watchers_count"]
            curid=repoData["id"]
            tmpData = CSVData(
                msg=curmsg, stars=curstars, forks=curforks,
                watchers=curwatchers, id=curid)
            totalCommitData.append(tmpData.getCSVRepresentation())

            #f.write(tmpData.getCSVRepresentation() + '\n')

        results[i] = totalCommitData;


    except Exception as e:
        print("ERROR IN THREADING CALL: ", repr(e))
        print(e.__traceback__());
        pass

f = io.open('linux-commits.csv', 'w', encoding='utf-8')
for repos in allUsersRepo:
    proxy = randomProxy()
    headers = randomUserAgent()

    r = requests.get(repos, proxies=proxy, headers=headers, timeout=5);
    print (r)
    jsonDataRepos = r.json();

    repoDataCount = 0
    for repoData in jsonDataRepos:
        repoDataCount+=1;

        #tmpData = CSVData(stars=repoData["stargazers_count"], forks=repoData["forks"], watchers=repoData["watchers_count"])
        cleanCommitUrl = repoData["commits_url"];
        cleanCommitUrl = cleanCommitUrl[0:cleanCommitUrl.index("{")]

        proxy = randomProxy()
        headers = randomUserAgent()
        try:
            commit_data_request = requests.get(cleanCommitUrl, proxies=proxy, headers=headers, timeout=5);
        except:
            continue;

        commit_data_json = commit_data_request.json();

        for commit_data in commit_data_json:
            try:
                tmpData = CSVData(msg=commit_data["commit"]["message"].replace('\n', ' ').replace('\r', '').replace(",",""), stars=repoData["stargazers_count"], forks=repoData["forks"],
                                  watchers=repoData["watchers_count"], id=repoData["id"])

                f.write(tmpData.getCSVRepresentation()+'\n')
            except:
                continue;



        try:
            pagesURLDirty = commit_data_request.headers["Link"]
            startPageClean = pagesURLDirty[pagesURLDirty.find("<")+1:pagesURLDirty.find(">")];
            end = pagesURLDirty.find(">");
            lastPageClean = pagesURLDirty[pagesURLDirty.find("<", end+1)+1:pagesURLDirty.find(">", end+1)]
            #print ("FIRST PAGE: ", startPageClean);
            startPageInt = int(startPageClean[-1])
            lastPageInt = int(lastPageClean.split("=")[1])
            print ("LAST PAGE NUMBER IS: " + str(lastPageInt))
            while (startPageInt <= lastPageInt):

                threads = [None] * THREAD_COUNT
                results = [None] * THREAD_COUNT

                for i in range(len(threads)):
                    threads[i] = Thread(target=fetchData, args=(startPageClean, repoData, i))
                    threads[i].start()
                    startPageInt += 1;
                    startPageClean = startPageClean.split("=")[0] + "=" + str(startPageInt)

                    if (startPageInt > lastPageInt):
                        break;


                for thread in threads:
                    if thread != None:
                        thread.join();


                for result in results:
                    if result != None:
                        for commit in result:
                            f.write(commit+'\n');

                # print ("scraping commit page: " + str(startPageInt))
                # proxy = randomProxy()
                # headers = randomUserAgent()
                # try :
                #     moreCommits = requests.get(startPageClean, proxies=proxy, headers=headers, timeout=5);
                #     commit_data_json = moreCommits.json();
                #
                #     for commit_data in commit_data_json:
                #         tmpData = CSVData(
                #             msg=commit_data["commit"]["message"].replace('\n', ' ').replace('\r', '').replace(",", ""),
                #             stars=repoData["stargazers_count"], forks=repoData["forks"],
                #             watchers=repoData["watchers_count"], id=repoData["id"])
                #
                #         f.write(tmpData.getCSVRepresentation()+'\n')
                #
                #     startPageInt += 1;
                #     startPageClean = startPageClean.split("=")[0] + "="+str(startPageInt)
                #     print("new page: " + startPageClean)
                #
                # except Exception as e:
                #     print (e)

        except UnicodeEncodeError:
            print ('something')
        except Exception as e:
            print("ERROR IN FETCH NEXT PG DATA: ", repr(e))
            continue;



f.close();