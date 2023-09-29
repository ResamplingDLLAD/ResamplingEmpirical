import Drain

data_name = 'BGL'
input_dir = '../../dataset/bgl/'  # The input directory of log file
output_dir = '../../dataset/bgl/'  # The output directory of parsing results
log_file = 'BGL.log'  # The input log file name

if data_name == 'BGL':
    log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
elif data_name == 'Thunderbird':
    log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component> <PID> <Content>'
elif data_name == 'Spirit':
    log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component> <PID> <Content>'  # log format

# Regular expression list for optional preprocessing (default: [])
regex = [
    r'blk_(|-)[0-9]+',  # block id
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
parser.parse(log_file)
