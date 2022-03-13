use rmp_serde;
use serde::Serialize;
use std::{
    io::Result,
    net::{SocketAddr, UdpSocket},
    thread,
    time::Duration,
};

pub struct UdpSender {
    socket: UdpSocket,
}

impl UdpSender {
    pub fn new(src_port: u16, dst_address: String) -> UdpSender {
        let socket = UdpSocket::bind(SocketAddr::from(([0, 0, 0, 0], src_port))).unwrap();
        // Block until hostname lookup succeeds
        while socket.connect(&dst_address).is_err() {
            thread::sleep(Duration::from_secs(3));
        }
        socket.set_nonblocking(true).unwrap();
        UdpSender { socket }
    }
    pub fn send(&self, msg: impl Serialize) -> Result<usize> {
        let buf = rmp_serde::to_vec(&msg).unwrap();
        self.socket.send(&buf)
    }
}
