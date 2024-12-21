package demo;

import a.b.c.Foo;

public class Hello {
  static Foo foo = new Foo();

  public static void main(String[] args) {
    String data = foo.bar(args[0]);
    Runtime.exec(data);
  }
}