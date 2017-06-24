import re
import random

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
allUsersRepo = ["https://api.github.com/users/google/repos"]
#commit message, repo-id, stars, forks, watchers, additions, deletions


f = open('csvdata3.txt', 'w')
for repos in allUsersRepo:
    proxy = randomProxy()
    headers = randomUserAgent()

    r = requests.get(repos, proxies=proxy, headers=headers, timeout=5);
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
            print ("FIRST PAGE: ", startPageClean);
            startPageInt = int(startPageClean[-1])
            lastPageInt = int(lastPageClean.split("=")[1])
            print ("LAST PAGE: " + str(lastPageInt))
            while (startPageInt <= lastPageInt):
                print ("scraping commit page: " + str(startPageInt))
                proxy = randomProxy()
                headers = randomUserAgent()
                try :
                    moreCommits = requests.get(startPageClean, proxies=proxy, headers=headers, timeout=5);
                    commit_data_json = commit_data_request.json();

                    for commit_data in commit_data_json:
                        tmpData = CSVData(
                            msg=commit_data["commit"]["message"].replace('\n', ' ').replace('\r', '').replace(",", ""),
                            stars=repoData["stargazers_count"], forks=repoData["forks"],
                            watchers=repoData["watchers_count"], id=repoData["id"])

                        f.write(tmpData.getCSVRepresentation()+'\n')

                except Exception as e:
                    print (e)

                startPageInt += 1;
                startPageClean = startPageClean[0:-1] + str(startPageInt)


        except Exception as e:
            print(e)
            continue;



f.close();