from crontab import CronTab
cron = CronTab(user='root')
job = cron.new(command='python fullprocess.py')
job.minute.every(1)
cron.write()