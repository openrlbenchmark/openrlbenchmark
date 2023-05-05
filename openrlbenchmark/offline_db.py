import peewee as pw
from playhouse.sqlite_ext import JSONField

database_proxy = pw.Proxy()  # Create a proxy for our db.

# db = pw.SqliteDatabase('runs_database.db')


class BaseModel(pw.Model):
    class Meta:
        database = database_proxy


class OfflineRun(BaseModel):
    id = pw.CharField()
    name = pw.CharField()
    state = pw.CharField()
    url = pw.CharField()
    path = JSONField()
    username = pw.CharField()
    project = pw.CharField()
    entity = pw.CharField()
    config = JSONField()


class Tag(BaseModel):
    name = pw.CharField()
    runs = pw.ManyToManyField(OfflineRun, backref="tags")


OfflineRunTag = Tag.runs.get_through_model()


if __name__ == "__main__":
    db_paths = ["db1.sqlite"]
    dbs = []
    Runs = []
    for db_path in db_paths:
        db = pw.SqliteDatabase(db_path)
        database_proxy.initialize(db)
        db.connect()
        db.create_tables([OfflineRun, Tag, OfflineRunTag])
        dbs.append(db)

    # db.connect()
    # db.create_tables([OfflineRun])
    tag1 = Tag.create(name="tag1")
    tag2 = Tag.create(name="tag2")
    for i in range(len(dbs)):
        with dbs[i].bind_ctx([OfflineRun]):
            new_run = OfflineRun.create(
                id="id",
                name="name",
                state="state",
                url="url",
                path=["path"],
                username="username",
                project="project",
                entity="entity",
                config={"config": "config"},
                tags=[tag1, tag2],
            )
            new_run.save()

    tags = ["tag1", "tag2"]
    for i in range(len(dbs)):
        with dbs[i].bind_ctx([OfflineRun]):
            # runs = OfflineRun.select()
            cond = True
            for tag in tags:
                cond = cond and (Tag.name == tag)
            runs = (
                OfflineRun.select()
                .join(OfflineRunTag)
                .join(Tag)
                .where(
                    # (OfflineRun.name == 'name') and
                    cond
                )
            )
            for run in runs:
                print(run.name)  # This will print a list of strings
