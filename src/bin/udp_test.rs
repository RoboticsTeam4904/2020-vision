use std::{env, thread::sleep, time::Duration};

use rand::Rng;
use stdvis_core::types::VisionTarget;
use vision_2020::udp::UdpSender;

fn main() {
    let mut args = env::args().skip(1);

    let dst_address = args.next().unwrap();
    let delay = args.next();

    let sender = UdpSender::new(0, dst_address); // 0 is random port

    if delay.is_some() {
        let delay: u64 = delay.unwrap().parse().unwrap();
        println!(
            "Sending packets every {} milliseconds. Press CTRL+C to quit.",
            delay
        );
        loop {
            send(&sender);
            sleep(Duration::from_millis(delay));
        }
    }
}

fn send(sender: &UdpSender) {
    let mut rng = rand::thread_rng();
    let test_vec = vec![VisionTarget {
        id: rng.gen(),
        beta: rng.gen(),
        theta: rng.gen(),
        dist: rng.gen(),
        height: rng.gen(),
        confidence: rng.gen(),
    }];
    sender.send(test_vec).unwrap();
}
