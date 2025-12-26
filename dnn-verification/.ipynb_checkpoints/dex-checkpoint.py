import csv
from collections import defaultdict

def parse_pack(packet):
    ether, ip = packet.split("|")[2:16], packet.split("|")[16:36]
    return {
        'source_mac': ":".join(ether[6:12]),
        'dest_mac': ":".join(ether[0:6]),
        'source_ip': ".".join(str(int(ip[12:16][i], 16)) for i in range(4)),
        'dest_ip': ".".join(str(int(ip[16:20][i], 16)) for i in range(4)),
        'protocol': ip[9],
        'packet_length': int(ip[2] + ip[3], 16),  # i.e. total length field in IP header
        'ttl': int(ip[8], 16),
        'payload_size': int(ip[2] + ip[3], 16) - (int(ip[0], 16) & 0x0F) * 4,  # i.e. total length - header length
        'source_port': int(ip[20] + ip[21], 16) if len(ip) > 21 and ip[9] in ['06', '11'] else 0,
        'dest_port': int(ip[22] + ip[23], 16) if len(ip) > 23 and ip[9] in ['06', '11'] else 0,
        'flags': int(ip[20], 16) & 0xFF if len(ip) > 20 and ip[9] == '06' else 0
    }

def generate(packets):
    ip_protocol_counts = defaultdict(lambda: defaultdict(int))
    for p in packets:
        ip_protocol_counts[(p['source_ip'], p['dest_ip'])][p['protocol']] += 1
    return [(src_ip, dest_ip, proto) for (src_ip, dest_ip), proto_counts in ip_protocol_counts.items()
            for proto, count in proto_counts.items() if count > 250]

def weigh_soul(source_ip, dest_ip, protocol, ruleset):
    return any(source_ip == rule[0] and dest_ip == rule[1] and protocol == rule[2] for rule in ruleset)

with open("data.txt", "r") as infile, open("data.csv", "w", newline="") as outfile:
    packets = [parse_pack(p) for p in infile.read().split("\n+---------+---------------+----------+") if p.strip()]
    ruleset = generate(packets)
    csv_writer = csv.writer(outfile)
    csv_writer.writerow([
        "Source MAC", "Destination MAC", "Source IP", "Destination IP", "Protocol", 
        "Packet Length", "TTL", "Payload Size", "Source Port", "Destination Port", "Flags", "Label"
    ])
    for p in packets:
        label = "malicious" if weigh_soul(p['source_ip'], p['dest_ip'], p['protocol'], ruleset) else "benign"
        csv_writer.writerow([
            p['source_mac'], p['dest_mac'], p['source_ip'], p['dest_ip'], p['protocol'],
            p['packet_length'], p['ttl'], p['payload_size'], p['source_port'], p['dest_port'], p['flags'], label
        ])
